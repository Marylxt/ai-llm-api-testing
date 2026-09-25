from faker import Faker
import csv
import time
from multiprocessing import Pool, cpu_count

TOTAL = 1_000_000
PROCESSES = cpu_count()

def generate_chunk(args):
    start_id, count, filename = args
    fake = Faker('zh_CN')
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for i in range(start_id, start_id + count):
            writer.writerow([
                i,
                fake.name(),
                fake.email(),
                fake.phone_number(),
                fake.city(),
                fake.company()
            ])
    return filename

if __name__ == '__main__':
    start = time.time()
    chunk_size = TOTAL // PROCESSES
    tasks = []
    for p in range(PROCESSES):
        start_id = p * chunk_size + 1
        count = chunk_size if p < PROCESSES - 1 else TOTAL - p * chunk_size
        tasks.append((start_id, count, f'users_part_{p}.csv'))

    with Pool(PROCESSES) as pool:
        files = pool.map(generate_chunk, tasks)

    # 合并文件
    with open('../users.csv', 'w', newline='', encoding='utf-8') as out:
        writer = csv.writer(out)
        writer.writerow(['id', 'name', 'email', 'phone', 'city', 'company'])
        for file in files:
            with open(file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)
                writer.writerows(reader)

    print(f'完成，总耗时 {time.time() - start:.2f} 秒')