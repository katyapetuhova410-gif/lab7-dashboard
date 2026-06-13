import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

class Dashboard:
    def __init__(self, root):
        self.root = root
        root.title('Офтальмологический дашборд')
        root.geometry('1000x700')
        root.configure(bg='#000000') # цвет фона черный
        root.resizable(False, False)

        self.df_raw = None # исходные данные (dataframe)
        self.df_work = None # рабочие данные после фильтров
        self.current_chart = "line" # текущий тип графика
        self.fig = plt.Figure(figsize=(8, 4.5), dpi=100)
        self.canvas = None

        # переменные для фильтров
        self.patient_var = tk.StringVar(value="Все") # выбранный пациент (pat_id)
        self.risk_var = tk.StringVar(value="Все") # выбранная категория риска
        self.time_var = tk.StringVar(value="Все") # выбранное время суток

        self.create_widgets()
        self.load_data()

    def load_data(self):
        try:
            # загрузка исходных данных из CSV
            self.df_raw = pd.read_csv('data.csv')
            # переименование столбцов как в варианте (ts, pat_id, iop, thick, acu, dp)
            self.df_raw.columns = ['ts', 'pat_id', 'iop', 'thick', 'acu', 'dp']

            # новые признаки (feature engineering)
            self.df_raw['date'] = pd.to_datetime(self.df_raw['ts'], unit='s') # дата осмотра
            self.df_raw['weekday'] = self.df_raw['date'].dt.dayofweek # день недели
            self.df_raw['hour'] = self.df_raw['date'].dt.hour # час осмотра

            # категория риска (на основе iop - внутриглазное давление)
            risk = []
            for v in self.df_raw['iop']:
                if v <= 15:
                    risk.append('Низкий')
                elif v <= 21:
                    risk.append('Средний')
                else:
                    risk.append('Высокий')
            self.df_raw['risk_category'] = risk

            # время суток (на основе часа)
            tod = []
            for h in self.df_raw['hour']:
                if 6 <= h < 12:
                    tod.append('Утро')
                elif 12 <= h < 18:
                    tod.append('День')
                elif 18 <= h < 22:
                    tod.append('Вечер')
                else:
                    tod.append('Ночь')
            self.df_raw['time_of_day'] = tod

            # очистка данных (как в лабе №6)
            self.df_raw['iop'] = np.clip(self.df_raw['iop'], 5, 60) # iop в [5,60]
            self.df_raw['thick'] = np.where(self.df_raw['thick'] < 0, 0, self.df_raw['thick']) # thick ≥ 0
            self.df_raw['acu'] = np.clip(self.df_raw['acu'], 0, 1) # acu в [0,1]

            self.df_work = self.df_raw.copy()
            self.update_status(f"Загружено записей: {len(self.df_raw):,} | Пациентов: {self.df_raw['pat_id'].nunique()}")
            self.update_filters()
            self.plot_line()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить data.csv\n{e}")

    def update_filters(self):
        # заполнение выпадающих списков
        patients = ["Все"] + sorted(self.df_raw['pat_id'].unique().tolist())
        self.patient_combo['values'] = patients
        self.risk_combo['values'] = ["Все", "Низкий", "Средний", "Высокий"]
        self.time_combo['values'] = ["Все", "Утро", "День", "Вечер", "Ночь"]

    def preprocess_data(self):
        # фильтрация данных перед отрисовкой
        self.df_work = self.df_raw.copy()
        # фильтр по pat_id (ID пациента)
        if self.patient_var.get() != "Все":
            self.df_work = self.df_work[self.df_work['pat_id'] == int(self.patient_var.get())]
        # фильтр по risk_category (категория риска)
        if self.risk_var.get() != "Все":
            self.df_work = self.df_work[self.df_work['risk_category'] == self.risk_var.get()]
        # фильтр по time_of_day (время суток)
        if self.time_var.get() != "Все":
            self.df_work = self.df_work[self.df_work['time_of_day'] == self.time_var.get()]
        return self.df_work

    def apply_filters(self):
        self.refresh_data() # применяем фильтры и обновляем график

    def clear_figure(self):
        self.fig.clear()

    # линейный график
    def plot_line(self):
        self.clear_figure()
        ax = self.fig.add_subplot(111)
        data = self.preprocess_data()

        if len(data) > 0:
            # группировка по дате, среднее iop
            grp = data.groupby('date')['iop'].mean().reset_index()
            ax.plot(grp['date'], grp['iop'], marker='o', linewidth=2, markersize=4, color='black', markerfacecolor='white')
            ax.set_xlabel('Дата')
            ax.set_ylabel('Среднее давление (мм рт.ст.)')
            ax.set_title('Динамика внутриглазного давления (iop)')
            ax.grid(True, alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
            ax.text(0.5, 0.5, 'Нет данных', ha='center', va='center', transform=ax.transAxes)

        self.fig.tight_layout()
        self.canvas.draw_idle()

    # столбчатая диаграмма
    def plot_bar(self):
        self.clear_figure()
        ax = self.fig.add_subplot(111)
        data = self.preprocess_data()

        if len(data) > 0:
            # группировка по категории риска, среднее iop
            grp = data.groupby('risk_category')['iop'].mean().reset_index()
            colors = ['green', 'orange', 'red']
            bars = ax.bar(grp['risk_category'], grp['iop'], color=colors, edgecolor='black')
            ax.set_xlabel('Категория риска')
            ax.set_ylabel('Среднее давление (мм рт.ст.)')
            ax.set_title('Среднее iop по категориям риска')

            # подписи значений на столбцах
            for bar, val in zip(bars, grp['iop']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}', ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, 'Нет данных', ha='center', va='center', transform=ax.transAxes)

        self.fig.tight_layout()
        self.canvas.draw_idle()

    # точечный график
    def plot_scatter(self):
        self.clear_figure()
        ax = self.fig.add_subplot(111)
        data = self.preprocess_data()

        if len(data) > 0:
            # iop vs thick, цвет = acu (острота зрения)
            sc = ax.scatter(data['iop'], data['thick'], c=data['acu'], cmap='viridis', alpha=0.6, s=30)
            ax.set_xlabel('Внутриглазное давление (iop), мм рт.ст.')
            ax.set_ylabel('Толщина роговицы (thick), мкм')
            ax.set_title('Зависимость thick от iop (цвет = acu)')
            plt.colorbar(sc, ax=ax, label='Острота зрения (acu)')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Нет данных', ha='center', va='center', transform=ax.transAxes)

        self.fig.tight_layout()
        self.canvas.draw_idle()

    # тепловая карта
    def plot_heatmap(self):
        self.clear_figure()
        ax = self.fig.add_subplot(111)
        data = self.preprocess_data()

        if len(data) > 0:
            # корреляция числовых признаков (iop, thick, acu, dp, weekday, hour)
            cols = ['iop', 'thick', 'acu', 'dp', 'weekday', 'hour']
            corr = data[cols].corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f', ax=ax, square=True)
            ax.set_title('Матрица корреляции признаков')
        else:
            ax.text(0.5, 0.5, 'Нет данных', ha='center', va='center', transform=ax.transAxes)

        self.fig.tight_layout()
        self.canvas.draw_idle()

    def refresh_data(self):
        # обновление текущего графика после изменения фильтров или типа
        if self.current_chart == "line":
            self.plot_line()
        elif self.current_chart == "bar":
            self.plot_bar()
        elif self.current_chart == "scatter":
            self.plot_scatter()
        elif self.current_chart == "heatmap":
            self.plot_heatmap()

    def export_plot(self):
        # сохранение текущего графика в файл
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("PDF", "*.pdf")])
        if path:
            self.fig.savefig(path, dpi=300, bbox_inches='tight')
            messagebox.showinfo("Экспорт", f"График сохранён: {path}")

    def update_status(self, msg):
        self.status_label.config(text=msg)
        self.root.update()

    def create_widgets(self):
        main = tk.Frame(self.root, bg='#000000')
        main.pack(padx=15, pady=15, fill=tk.BOTH, expand=True)

        tk.Label(main, text='ОФТАЛЬМОЛОГИЧЕСКИЙ ДАШБОРД', font=('Courier New', 14, 'bold'), fg='#c0c0c0', bg='#000000').pack(pady=10)

        # ПАНЕЛЬ КНОПОК 
        btn_frame = tk.LabelFrame(main, text='Тип графика', font=('Courier New', 9), fg='#a0a0a0', bg='#0a0a0a')
        btn_frame.pack(fill=tk.X, pady=5)

        inner = tk.Frame(btn_frame, bg='#0a0a0a')
        inner.pack(pady=8)

        # кнопки переключения типов графиков
        tk.Button(inner, text='Линейный', command=lambda: self._set_chart("line"), width=12, font=('Courier New', 9), bg='#1a1a1a', fg='#c0c0c0').pack(side=tk.LEFT, padx=4)
        tk.Button(inner, text='Столбчатый', command=lambda: self._set_chart("bar"), width=12, font=('Courier New', 9), bg='#1a1a1a', fg='#c0c0c0').pack(side=tk.LEFT, padx=4)
        tk.Button(inner, text='Точечный', command=lambda: self._set_chart("scatter"), width=12, font=('Courier New', 9), bg='#1a1a1a', fg='#c0c0c0').pack(side=tk.LEFT, padx=4)
        tk.Button(inner, text='Тепловая карта', command=lambda: self._set_chart("heatmap"), width=14, font=('Courier New', 9), bg='#1a1a1a', fg='#c0c0c0').pack(side=tk.LEFT, padx=4)

        tk.Button(inner, text='Экспорт', command=self.export_plot, width=10, font=('Courier New', 9), bg='#1a1a1a', fg='#c0c0c0').pack(side=tk.RIGHT, padx=4)
        tk.Button(inner, text='Обновить', command=self.refresh_data, width=10, font=('Courier New', 9), bg='#1a1a1a', fg='#c0c0c0').pack(side=tk.RIGHT, padx=4)

        # ПАНЕЛЬ ФИЛЬТРОВ 
        f_frame = tk.LabelFrame(main, text='Фильтры', font=('Courier New', 9), fg='#a0a0a0', bg='#0a0a0a')
        f_frame.pack(fill=tk.X, pady=5)

        f_inner = tk.Frame(f_frame, bg='#0a0a0a')
        f_inner.pack(pady=8, padx=10)

        # фильтр по pat_id (ID пациента)
        tk.Label(f_inner, text='Пациент:', font=('Courier New', 9), fg='#c0c0c0', bg='#0a0a0a').pack(side=tk.LEFT, padx=5)
        self.patient_combo = ttk.Combobox(f_inner, textvariable=self.patient_var, width=10, state="readonly")
        self.patient_combo.pack(side=tk.LEFT, padx=5)

        # фильтр по risk_category (категория риска)
        tk.Label(f_inner, text='Риск:', font=('Courier New', 9), fg='#c0c0c0', bg='#0a0a0a').pack(side=tk.LEFT, padx=5)
        self.risk_combo = ttk.Combobox(f_inner, textvariable=self.risk_var, width=12, state="readonly")
        self.risk_combo.pack(side=tk.LEFT, padx=5)

        # фильтр по time_of_day (время суток)
        tk.Label(f_inner, text='Время суток:', font=('Courier New', 9), fg='#c0c0c0', bg='#0a0a0a').pack(side=tk.LEFT, padx=5)
        self.time_combo = ttk.Combobox(f_inner, textvariable=self.time_var, width=12, state="readonly")
        self.time_combo.pack(side=tk.LEFT, padx=5)

        # кнопка применения всех фильтров
        tk.Button(f_inner, text='Применить', command=self.apply_filters, width=12, font=('Courier New', 9), bg='#1a1a1a', fg='#c0c0c0').pack(side=tk.RIGHT, padx=5)

        # ОБЛАСТЬ ГРАФИКА 
        plot_frame = tk.LabelFrame(main, text='График', font=('Courier New', 9), fg='#a0a0a0', bg='#0a0a0a')
        plot_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # встраивание графика matplotlib в Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # СТРОКА СОСТОЯНИЯ 
        status_frame = tk.Frame(main, bg='#000000')
        status_frame.pack(fill=tk.X, pady=5)

        self.status_label = tk.Label(status_frame, text='Загрузка...', font=('Courier New', 8), fg='#808080', bg='#000000')
        self.status_label.pack(side=tk.LEFT)

        # отключаем авто-обновление при изменении фильтров (обновление только по кнопке)
        self.patient_var.trace('w', lambda *_: None)
        self.risk_var.trace('w', lambda *_: None)
        self.time_var.trace('w', lambda *_: None)

    def _set_chart(self, chart_type):
        self.current_chart = chart_type
        self.refresh_data()

def main():
    root = tk.Tk()
    Dashboard(root)
    root.mainloop()

if __name__ == '__main__':
    main()