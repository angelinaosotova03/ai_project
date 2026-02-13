with open('last_observations.csv', 'r', encoding='utf-8-sig') as f:
    content = f.read()
    print(f"Длина содержимого: {len(content)} символов")
    print("Первые 200 символов:")
    print(repr(content[:200]))  # Покажет скрытые символы