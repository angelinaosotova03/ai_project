# -*- coding: utf-8 -*-
"""
Flask сервер для прогнозирования добычи газа
Полная интеграция с моделью и веб-интерфейсом
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from forecast_backend import generate_forecast_from_csv
import matplotlib
matplotlib.use('Agg')  # Для работы без GUI
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__, static_folder='templates')
CORS(app)  # Разрешаем CORS для всех запросов

# Настройки
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 МБ
UPLOAD_FOLDER = 'uploads'
DOWNLOAD_FOLDER = 'downloads'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

print("="*70)
print("🚀 ЗАПУСК СЕРВЕРА ПРОГНОЗИРОВАНИЯ ДОБЫЧИ ГАЗА")
print("="*70)

@app.route('/')
def index():
    """Главная страница"""
    return send_from_directory('templates', 'forecast_interface.html')

@app.route('/<path:path>')
def static_proxy(path):
    """Обслуживание статических файлов"""
    return send_from_directory('templates', path)

@app.route('/api/forecast', methods=['POST'])
def forecast():
    """ГЕНЕРАЦИЯ ПРОГНОЗА С ГРАФИКАМИ"""
    try:
        print("\n" + "="*70)
        print("📡 ПОЛУЧЕН ЗАПРОС НА ПРОГНОЗ")
        print("="*70)
        
        data = request.get_json()
        
        if not data:
            print("❌ Ошибка: нет данных в запросе")
            return jsonify({'success': False, 'message': 'Нет данных в запросе'}), 400
        
        well_name = data.get('wellName')
        forecast_days = int(data.get('forecastDays', 30))
        csv_data = data.get('csvData')
        
        print(f"Скважина: {well_name}")
        print(f"Горизонт: {forecast_days} дней")
        
        if not well_name:
            return jsonify({'success': False, 'message': 'Укажите название скважины'}), 400
        
        if forecast_days < 7 or forecast_days > 90:
            return jsonify({'success': False, 'message': 'Горизонт прогноза: 7-90 дней'}), 400
        
        # Создание DataFrame
        if not csv_data or 'data' not in csv_data or 'headers' not in csv_data:
            return jsonify({'success': False, 'message': 'Неверный формат данных CSV'}), 400
        
        df = pd.DataFrame(csv_data['data'], columns=csv_data['headers'])
        print(f"Загружено {len(df)} строк")
        print(f"Колонки: {df.columns.tolist()}")
        
        # Переименование колонок
        column_mapping = {
            'Gas Volume (MMscf)': 'Gas_Volume',
            'Oil Volume (stb)': 'Oil_Volume',
            'Water Volume (stb)': 'Water_Volume',
            'Water Volume  (stb)': 'Water_Volume',
            'Time (Days)': 'Time',
            'Choke Size': 'Choke_Size',
            'Gas Lift Inj Volume  (MMscf)': 'Gas_Lift_Inj_Volume',
            'Pressure Source': 'Pressure_Source',
            'Calculated Sandface Pressure  (psi(a))': 'Sandface_Pressure'
        }
        
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
                print(f"Переименована колонка: {old_name} → {new_name}")
        
        # Переименование колонок давления
        pressure_mapping = {
            'Casing Pressure  (psi(a))': 'Casing_Pressure',
            'Tubing Pressure  (psi(a))': 'Tubing_Pressure',
            'Active Pressure  (psi(a))': 'Active_Pressure',
            'Line Pressure  (psi(a))': 'Line_Pressure'
        }
        
        for old_name, new_name in pressure_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
                print(f"Переименована колонка давления: {old_name} → {new_name}")
        
        if 'Lease' not in df.columns:
            return jsonify({
                'success': False,
                'message': f'Колонка "Lease" не найдена. Доступны: {df.columns.tolist()}'
            }), 400
        
        well_data = df[df['Lease'] == well_name].copy()
        print(f"Данных для скважины {well_name}: {len(well_data)} записей")
        
        if len(well_data) < 10:
            return jsonify({
                'success': False,
                'message': f'Недостаточно данных для скважины {well_name} ({len(well_data)} записей)'
            }), 400
        
        # Преобразование числовых колонок
        numeric_cols_to_convert = ['Time', 'Gas_Volume', 'Oil_Volume', 'Water_Volume', 
                                  'Choke_Size', 'Casing_Pressure', 'Tubing_Pressure']
        for col in numeric_cols_to_convert:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                well_data[col] = pd.to_numeric(well_data[col], errors='coerce')
        
        # === ЗАГРУЗКА МОДЕЛИ С ЭНКОДЕРОМ СКВАЖИН ===
        if not os.path.exists('gas_forecast_model.pkl'):
            print("❌ Ошибка: модель не найдена")
            return jsonify({
                'success': False,
                'message': 'Модель не найдена! Обучите модель: python train_model_all_wells.py'
            }), 500

        print("   Загрузка модели...")
        try:
            saved = joblib.load('gas_forecast_model.pkl')
            model = saved['model']
            scaler = saved['scaler']
            imputer = saved['imputer']
            feature_cols = saved['feature_cols']
            lease_encoder = saved.get('lease_encoder')  # КРИТИЧЕСКИ ВАЖНО: получаем энкодер!
            
            print(f"   ✅ Модель загружена. Признаков: {len(feature_cols)}")
            if lease_encoder:
                print(f"   ✅ Энкодер скважин загружен ({len(lease_encoder.classes_)} скважин)")
            else:
                print("   ⚠️  Энкодер скважин отсутствует в модели (возможно, старая версия)")
                
        except Exception as e:
            import traceback
            print("❌ Ошибка загрузки модели:")
            traceback.print_exc()
            return jsonify({
                'success': False,
                'message': f'Ошибка загрузки модели: {str(e)}. Переобучите модель: python train_model_all_wells.py'
            }), 500

        # === ГЕНЕРАЦИЯ ПРОГНОЗА С ЭНКОДЕРОМ ===
        print("   🧠 Генерация прогноза через модель...")
        try:
            forecast_df = generate_forecast_from_csv(
                well_data=well_data,
                well_name=well_name,
                forecast_days=forecast_days,
                model=model,
                scaler=scaler,
                imputer=imputer,
                feature_cols=feature_cols,
                lease_encoder=lease_encoder  # ПЕРЕДАЁМ ЭНКОДЕР!
            )
        except Exception as e:
            import traceback
            print("❌ Ошибка в generate_forecast_from_csv:")
            traceback.print_exc()
            return jsonify({
                'success': False,
                'message': f'Ошибка генерации прогноза: {str(e)}'
            }), 500
        # === ГЕНЕРАЦИЯ ГРАФИКОВ ===
        print("Генерация графиков...")
        charts = generate_charts(well_data, forecast_df, well_name, model, feature_cols)
        
        # Подготовка результатов
        result_data = forecast_df.to_dict('records')
        stats = {
            'well_name': well_name,
            'forecast_days': forecast_days,
            'avg_production': round(forecast_df['gas'].mean(), 4),
            'total_volume': round(forecast_df['gas'].sum(), 2),
            'start_day': int(forecast_df['day'].iloc[0]),
            'end_day': int(forecast_df['day'].iloc[-1]),
            'forecast_data': result_data,
            'charts': charts  # Добавляем графики в ответ
        }
        
        print("="*70)
        print("✅ ПРОГНОЗ И ГРАФИКИ УСПЕШНО СГЕНЕРИРОВАНЫ")
        print(f"Средняя добыча: {stats['avg_production']:.4f} MMscf/день")
        print(f"Суммарный объём: {stats['total_volume']:.2f} MMscf")
        print("="*70 + "\n")
        
        return jsonify({
            'success': True,
            'message': 'Прогноз успешно сгенерирован с визуализацией',
            'stats': stats
        })
        
    except Exception as e:
        import traceback
        print("\n" + "="*70)
        print("❌ КРИТИЧЕСКАЯ ОШИБКА")
        print("="*70)
        traceback.print_exc()
        print("="*70 + "\n")
        
        return jsonify({
            'success': False,
            'message': f'Ошибка сервера: {str(e)}'
        }), 500

def generate_charts(well_data, forecast_df, well_name, model, feature_cols):
    """Генерация 3 профессиональных графиков в формате base64"""
    charts = {}
    
    # Настройка стиля
    plt.rcParams['font.family'] = 'Segoe UI'
    sns.set_style("whitegrid")
    
    # === ГРАФИК 1: История добычи + Прогноз ===
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    # Исторические данные (последние 60 дней)
    history = well_data.tail(60).copy()
    history['Time'] = pd.to_numeric(history['Time'], errors='coerce')
    history['Gas_Volume'] = pd.to_numeric(history['Gas_Volume'], errors='coerce')
    history = history.dropna(subset=['Time', 'Gas_Volume'])
    
    if len(history) > 0:
        ax1.plot(history['Time'], history['Gas_Volume'], 
                'o-', color='#1a3a6c', label='История добычи', 
                linewidth=2.5, markersize=6, alpha=0.8)
    
    # Прогноз
    ax1.plot(forecast_df['day'], forecast_df['gas'], 
            's--', color='#e63946', label=f'Прогноз ({len(forecast_df)} дней)', 
            linewidth=3, markersize=8, alpha=0.9)
    
    # Вертикальная линия разделения
    if len(history) > 0:
        ax1.axvline(x=history['Time'].max(), color='gray', linestyle='--', 
                   alpha=0.5, linewidth=2, label='Текущий день')
    
    ax1.set_xlabel('Время (дни)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Добыча газа (MMscf)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Добыча газа — скважина {well_name}', fontsize=14, fontweight='bold', pad=20)
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#f8f9fa')
    
    # Сохранение в base64
    buf1 = io.BytesIO()
    plt.savefig(buf1, format='png', dpi=150, bbox_inches='tight', facecolor='#f8f9fa')
    buf1.seek(0)
    charts['production_forecast'] = base64.b64encode(buf1.read()).decode('utf-8')
    plt.close(fig1)
    
    # === ГРАФИК 2: Тренд добычи и накопленный объём ===
    fig2, (ax21, ax22) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Тренд добычи
    ax21.plot(forecast_df['day'], forecast_df['gas'], 
             color='#2a9d8f', linewidth=3, marker='o', markersize=6)
    ax21.fill_between(forecast_df['day'], forecast_df['gas'], alpha=0.3, color='#2a9d8f')
    ax21.set_xlabel('День прогноза', fontsize=11, fontweight='bold')
    ax21.set_ylabel('Добыча (MMscf/день)', fontsize=11, fontweight='bold')
    ax21.set_title('Тренд добычи', fontsize=12, fontweight='bold', pad=15)
    ax21.grid(True, alpha=0.3)
    
    # Накопленный объём
    cumulative = forecast_df['gas'].cumsum()
    ax22.plot(forecast_df['day'], cumulative, 
             color='#1a3a6c', linewidth=3, marker='s', markersize=6)
    ax22.fill_between(forecast_df['day'], cumulative, alpha=0.2, color='#1a3a6c')
    ax22.set_xlabel('День прогноза', fontsize=11, fontweight='bold')
    ax22.set_ylabel('Накопленный объём (MMscf)', fontsize=11, fontweight='bold')
    ax22.set_title(f'Накоплено: {cumulative.iloc[-1]:.1f} MMscf', 
                  fontsize=12, fontweight='bold', pad=15)
    ax22.grid(True, alpha=0.3)
    
    plt.tight_layout()
    buf2 = io.BytesIO()
    plt.savefig(buf2, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf2.seek(0)
    charts['trend_cumulative'] = base64.b64encode(buf2.read()).decode('utf-8')
    plt.close(fig2)
    
    # === ГРАФИК 3: Важность признаков (из модели sklearn) ===
    if hasattr(model, 'feature_importances_'):
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        
        # Получение важности признаков
        importances = model.feature_importances_
        indices = importances.argsort()[::-1][:10]  # Топ-10 признаков
        
        # Подготовка данных
        top_features = [feature_cols[i] for i in indices]
        top_importances = importances[indices]
        
        # Горизонтальная барчарт
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
        bars = ax3.barh(range(len(top_features)), top_importances, color=colors)
        
        # Настройка осей
        ax3.set_yticks(range(len(top_features)))
        ax3.set_yticklabels([f.replace('_', ' ').title() for f in top_features], fontsize=11)
        ax3.set_xlabel('Важность признака', fontsize=12, fontweight='bold')
        ax3.set_title('Важность признаков для прогноза (sklearn)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax3.invert_yaxis()  # Самый важный сверху
        
        # Добавление значений на бары
        for i, (bar, imp) in enumerate(zip(bars, top_importances)):
            ax3.text(imp + 0.01, i, f'{imp:.3f}', 
                    va='center', fontsize=10, fontweight='bold')
        
        ax3.grid(axis='x', alpha=0.3)
        ax3.set_facecolor('#f8f9fa')
        
        plt.tight_layout()
        buf3 = io.BytesIO()
        plt.savefig(buf3, format='png', dpi=150, bbox_inches='tight', facecolor='#f8f9fa')
        buf3.seek(0)
        charts['feature_importance'] = base64.b64encode(buf3.read()).decode('utf-8')
        plt.close(fig3)
    
    return charts

@app.route('/api/health', methods=['GET'])
def health_check():
    """Проверка работоспособности сервера"""
    model_exists = os.path.exists('gas_forecast_model.pkl')
    return jsonify({
        'status': 'ok',
        'server': 'running',
        'model_loaded': model_exists,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("📌 Сервер запущен на: http://localhost:5001")
    print("📌 Максимальный размер файла: 16 МБ")
    print("📌 Поддерживаемые форматы: .csv")
    print("="*70)
    print("\n💡 ВАЖНО: Открывайте интерфейс через браузер по адресу:")
    print("   http://localhost:5001")
    print("   (НЕ открывайте файл напрямую через двойной клик!)\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)