from faker import Faker
import csv
import time

fake = Faker('zh_CN')
TOTAL = 1_000_000
BATCH = 10000

start = time.time()
with open('../users.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'name', 'email', 'phone', 'city', 'company'])
    for i in range(1, TOTAL + 1):
        writer.writerow([
            i,
            fake.name(),
            fake.email(),
            fake.phone_number(),
            fake.city(),
            fake.company()
        ])
        if i % BATCH == 0:
            print(f'已生成 {i} 条，耗时 {time.time() - start:.1f} 秒')

print(f'完成，总耗时 {time.time() - start:.2f} 秒')