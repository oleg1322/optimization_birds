import sys
import logging
from collections import defaultdict
import heapq
from typing import List, Tuple, Dict, Optional

# ====================== ЛОГИРОВАНИЕ ======================
log_filename = "birds.log"
open(log_filename, 'w').close()

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ====================== ПАРСИНГ ======================
def parse_data(lines: List[str]) -> Tuple[List[List[str]], int]:
    data = []
    in_data = False
    for line in lines:
        line = line.strip()
        if not line or line.startswith('--'): continue
        if line == 'DATA':
            in_data = True
            logger.info("Найден блок DATA")
            continue
        if line == '/':
            if in_data: break
            continue
        if in_data:
            if line == '==': data.append([])
            else: data.append(line.split())
    
    if not data:
        logger.error("Блок DATA не найден")
        return [], 0
    
    N = len(data[0]) if data[0] else 0
    non_empty = [c for c in data if c]
    if non_empty and any(len(c) != N for c in non_empty):
        logger.error("Не все непустые колонки имеют одинаковую длину")
        raise ValueError("All non-empty columns must have same length")
    
    logger.info(f"Распаршено {len(data)} колонок, высота N={N}")
    return data, N

# ====================== СОСТОЯНИЕ ======================
State = Tuple[Tuple[str, ...], ...]  # Полное состояние

def state_to_tuple(board: List[List[str]]) -> State:
    return tuple(tuple(col) for col in board)

def is_solved(board: List[List[str]], N: int) -> bool:
    return all((not col or (len(col) == N and len(set(col)) == 1)) for col in board)

def count_complete(board: List[List[str]], N: int) -> int:
    return sum(1 for col in board if col and len(col) == N and len(set(col)) == 1)

# ====================== ЭВРИСТИКА ======================
def heuristic(board: List[List[str]], N: int, targets: Dict[str, int]) -> int:
    score = 0
    complete = count_complete(board, N)
    score -= complete * 10000

    # Птицы на месте снизу
    for col in board:
        if not col: continue
        bottom = col[0]
        matched = 0
        for b in col:
            if b == bottom:
                matched += 1
            else:
                break
        score -= matched * 50

    # Штраф за мусор
    for col in board:
        if not col or len(set(col)) == 1: continue
        for i in range(len(col)):
            if col[i] != col[0]:
                score += (len(col) - i) * 5
                break

    # Бонус за пустые
    empty = sum(1 for col in board if not col)
    score -= empty * 20

    # Бонус за почти полные
    almost = sum(1 for col in board if col and len(set(col)) == 1 and len(col) >= N - 2)
    score -= almost * 100

    return score

# ====================== ПОИСК ======================
Move = Tuple[int, int]

class Solver:
    def __init__(self, board: List[List[str]], N: int, beam_width: int = 500, max_states: int = 1000000):
        self.N = N
        self.num_cols = len(board)
        self.initial = [col[:] for col in board]
        self.targets = self.count_birds()
        self.beam_width = beam_width
        self.max_states = max_states

    def count_birds(self) -> Dict[str, int]:
        count = defaultdict(int)
        for col in self.initial:
            for b in col: count[b] += 1
        for b, c in count.items():
            if c % self.N != 0:
                logger.error(f"{b}: {c} не делится на {self.N}")
                raise ValueError("Dividing fail")
        return count

    def get_moves(self, board: List[List[str]]) -> List[Move]:
        moves = []
        for i in range(self.num_cols):
            if not board[i]: continue
            bird = board[i][-1]
            for j in range(self.num_cols):
                if i == j or len(board[j]) >= self.N: continue
                if not board[j] or board[j][-1] == bird:
                    moves.append((i, j))
        return moves

    def solve(self) -> Optional[List[Move]]:
        start_state = state_to_tuple(self.initial)
        frontier = [(heuristic(self.initial, self.N, self.targets), 0, start_state, [])]
        visited = {start_state: 0}
        explored = 0
        logger.info("Запуск A* с Beam Search...")

        while frontier:
            _, cost, state, path = heapq.heappop(frontier)
            board = [list(col) for col in state]
            explored += 1

            if explored % 1000 == 0:
                logger.info(f"Прогресс: исследовано {explored}, очередь: {len(frontier)}, ходов: {cost}")

            if is_solved(board, self.N):
                L = count_complete(board, self.N)
                F = 100 * self.N * L - len(path)
                logger.info(f"РЕШЕНИЕ НАЙДЕНО! Ходов: {len(path)}, L={L}, F={F}")
                return path

            if len(visited) > self.max_states:
                logger.warning("Превышено максимальное число состояний.")
                break

            for i, j in self.get_moves(board):
                new_board = [col[:] for col in board]
                bird = new_board[i].pop()
                new_board[j].append(bird)
                new_state = state_to_tuple(new_board)
                new_cost = cost + 1

                if new_state not in visited or new_cost < visited[new_state]:
                    visited[new_state] = new_cost
                    h = heuristic(new_board, self.N, self.targets)
                    priority = new_cost + h
                    heapq.heappush(frontier, (priority, new_cost, new_state, path + [(i, j)]))

            # Beam
            if len(frontier) > self.beam_width:
                frontier = heapq.nsmallest(self.beam_width, frontier)
                heapq.heapify(frontier)

        logger.warning("Решение не найдено.")
        return None

# ====================== MAIN ======================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="ПТИЦЫ 2025 — Решатель")
    parser.add_argument('input_file')
    parser.add_argument('--output', '-o')
    parser.add_argument('--beam', type=int, default=500, help='Beam width')
    parser.add_argument('--max_states', type=int, default=1000000, help='Max visited states')
    args = parser.parse_args()

    logger.info(f"Входной файл: {args.input_file}")

    with open(args.input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    board, N = parse_data(lines)
    if not board: return

    solver = Solver(board, N, beam_width=args.beam)
    solution = solver.solve()

    if not solution:
        print("Решение не найдено")
        return

    sim = [col[:] for col in board]
    order_lines = []
    for idx, (fr, to) in enumerate(solution, 1):
        bird = sim[fr][-1]
        sim[fr].pop()
        sim[to].append(bird)
        order_lines.append(f"{idx} {fr+1} {to+1} {bird}")

    L = count_complete(sim, N)
    K = len(solution)
    F = 100 * N * L - K

    logger.info("====")
    logger.info("перестановки")
    logger.info("====")
    for line in order_lines:
        logger.info(line)
    logger.info(f"ГОТОВО! F = {F}")

    output = ["DATA"]
    for col in board:
        output.append("==" if not col else " ".join(col))
    output.extend(["==", "==", "==", "/"])
    output.append("ORDER")
    for line in order_lines:
        parts = line.split()[1:]
        output.append(" ".join(parts))
    output.extend(["/", f"K = {K}, L = {L}, F = {F}"])

    result = "\n".join(output)
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(result)
        logger.info(f"Сохранено в {args.output}")
    else:
        print(result)

if __name__ == "__main__":
    main()
