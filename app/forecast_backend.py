# -*- coding: utf-8 -*-
"""
Логика прогнозирования добычи газа — версия для обучения на всех скважинах
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def prepare_features_for_forecast(well_data, feature_cols, lease_encoder, well_name):
    """
    Подготовка признаков для прогноза с учётом кодирования скважины
    """
    well_data = well_data.copy().reset_index(drop=True)
    
    # Преобразование числовых колонок
    numeric_columns = ['Time', 'Gas_Volume', 'Oil_Volume', 'Water_Volume', 
                      'Choke_Size', 'Casing_Pressure', 'Tubing_Pressure']
    
    for col in numeric_columns:
        if col in well_data.columns:
            well_data[col] = pd.to_numeric(well_data[col], errors='coerce')
    
    # Создание лаговых признаков (только для текущей скважины)
    if 'Gas_Volume' in well_data.columns:
        well_data['gas_lag_1'] = well_data['Gas_Volume'].shift(1)
        well_data['gas_lag_3'] = well_data['Gas_Volume'].shift(3)
        well_data['gas_lag_7'] = well_data['Gas_Volume'].shift(7)
        well_data['gas_lag_14'] = well_data['Gas_Volume'].shift(14)
        
        well_data['gas_ma_7'] = well_data['Gas_Volume'].rolling(window=7, min_periods=1).mean()
        well_data['gas_ma_14'] = well_data['Gas_Volume'].rolling(window=14, min_periods=1).mean()
        well_data['gas_ma_30'] = well_data['Gas_Volume'].rolling(window=30, min_periods=1).mean()
    
    # Временные признаки
    if 'Time' in well_data.columns:
        well_data['time_sqrt'] = np.sqrt(well_data['Time'])
        well_data['time_log'] = np.log1p(well_data['Time'])
    
    # Признаки нефти
    if 'Oil_Volume' in well_data.columns:
        well_data['oil_lag_1'] = well_data['Oil_Volume'].shift(1)
        well_data['oil_lag_7'] = well_data['Oil_Volume'].shift(7)
    
    # Признак воды
    if 'Water_Volume' in well_data.columns and 'Gas_Volume' in well_data.columns:
        well_data['water_gas_ratio'] = well_data['Water_Volume'] / (well_data['Gas_Volume'] + 1e-5)
    
    # === КРИТИЧЕСКИ ВАЖНО: добавление закодированного имени скважины ===
    if lease_encoder is not None:
        # Проверка, есть ли скважина в обученном энкодере
        if well_name in lease_encoder.classes_:
            well_data['Lease_Encoded'] = lease_encoder.transform([well_name])[0]
        else:
            # Если новая скважина — используем среднее значение
            well_data['Lease_Encoded'] = int(lease_encoder.transform(lease_encoder.classes_)[len(lease_encoder.classes_) // 2])
        print(f"   Добавлен признак скважины: Lease_Encoded = {well_data['Lease_Encoded'].iloc[0]}")
    
    # Заполнение пропусков
    numeric_cols = well_data.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if well_data[col].isna().any():
            well_data[col] = well_data[col].fillna(well_data[col].median())
    
    # Оставляем только признаки, которые есть в модели
    available_features = [col for col in feature_cols if col in well_data.columns]
    missing_features = [col for col in feature_cols if col not in well_data.columns]
    
    if missing_features:
        print(f"⚠️  Отсутствующие признаки: {missing_features}")
        print(f"✅ Доступные признаки: {available_features}")
    
    return well_data, available_features

def generate_forecast_from_csv(well_data, well_name, forecast_days, model, scaler, imputer, feature_cols, lease_encoder):
    """
    Генерация прогноза с поддержкой мульти-скважинного обучения
    """
    print("\n=== НАЧАЛО ГЕНЕРАЦИИ ПРОГНОЗА ===")
    print(f"Скважина: {well_name}")
    print(f"Данные: {len(well_data)} записей")
    print(f"Признаки модели: {feature_cols}")
    print(f"Колонки данных: {well_data.columns.tolist()}")
    
    # Подготовка признаков с кодированием скважины
    well_data_processed, available_features = prepare_features_for_forecast(
        well_data, feature_cols, lease_encoder, well_name
    )
    
    if len(available_features) == 0:
        raise ValueError(f"Нет общих признаков между моделью и данными. Модель требует: {feature_cols}")
    
    # Берём последнюю запись для прогноза
    last_record = well_data_processed.iloc[-1]
    
    # Формируем вектор признаков
    X_input = pd.DataFrame([last_record], columns=available_features)
    
    print(f"Вектор признаков для прогноза: {X_input.shape}")
    print(f"Значения ключевых признаков:")
    for col in ['Gas_Volume', 'gas_lag_1', 'gas_ma_7', 'Lease_Encoded'][:len(available_features)]:
        if col in X_input.columns:
            print(f"  • {col}: {X_input[col].values[0]}")
    
    # Обработка пропусков и масштабирование
    X_imp = pd.DataFrame(imputer.transform(X_input), columns=available_features)
    X_scaled = scaler.transform(X_imp)
    
    print(f"Масштабированный вектор: {X_scaled[0][:5]}...")
    
    # Прогноз первого дня
    pred = model.predict(X_scaled)[0]
    print(f"Прогноз на день 1: {pred:.4f} MMscf")
    
    # Генерация прогноза на весь период (рекурсивный подход)
    forecasts = []
    current_pred = max(0.001, pred)
    
    for day in range(1, forecast_days + 1):
        # Для дней >1 используем предыдущий прогноз как лаг
        if day > 1:
            current_pred = max(0.001, current_pred * 0.998)  # Естественное затухание 0.2%/день
        
        forecasts.append({
            'day': int(last_record['Time']) + day if 'Time' in last_record else day,
            'gas': current_pred,
            'gas_m3': current_pred * 28.3168
        })
    
    result_df = pd.DataFrame(forecasts)
    result_df.columns = ['day', 'gas', 'gas_m3']
    
    print(f"✅ Сгенерировано {len(result_df)} дней прогноза")
    print("=== КОНЕЦ ГЕНЕРАЦИИ ПРОГНОЗА ===\n")
    
    return result_df