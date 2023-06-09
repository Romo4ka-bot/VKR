# Standard python libraries
import os
import random
import warnings
from collections import defaultdict
from functools import wraps
from typing import List, Tuple, Dict, Any, Optional, Union, Iterable, Callable

import matplotlib.pyplot as plt
import numpy as np
# Essential DS libraries
import pandas as pd
# For visualization
import seaborn as sns
from catboost import CatBoostRegressor, Pool
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold


class CatBoostPipeline:
    def __init__(self, categorical_features: List[str], numerical_features: List[str], date_features: List[str]):
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.date_features = date_features

    def preprocess_data(self, data: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        X = data[self.numerical_features + self.categorical_features]

        if target_column != '':
            y = data[target_column]
        else:
            y = []

        # Вычисление новых признаков из дат
        for date_col in self.date_features:
            X[f'days_since_{date_col}'] = (data['Date Last Contact CVRM'] - data[date_col]).dt.days

        # Замена пропущенных значений
        X[self.numerical_features].fillna(-9999, inplace=True)

        return X, y

    def hyperparameter_tuning(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 4):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        cat_features_indices = [X.columns.get_loc(col) for col in self.categorical_features]

        train_pool = Pool(data=X_train, label=y_train, cat_features=cat_features_indices)
        val_pool = Pool(data=X_val, label=y_val, cat_features=cat_features_indices)

        # Настройка параметров для подбора
        params = {'iterations': 1000,
                  'loss_function': 'RMSE',
                  'random_seed': 42}

        # Подбор гиперпараметров
        model = CatBoostRegressor(**params, verbose=False)
        grid = {'learning_rate': [0.03, 0.1],
                'depth': [4, 6, 10],
                'l2_leaf_reg': [1, 3, 5, 7, 9]}

        grid_search_result = model.grid_search(grid, train_pool)

        # Сохранение модели
        model.save_model("best_model.cbm")

        return grid_search_result['params'], grid_search_result['cv_results']['test-RMSE-mean'][-1]

    def run_cross_validation(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 4):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        cat_features_indices = [X.columns.get_loc(col) for col in self.categorical_features]

        model = CatBoostRegressor(cat_features=cat_features_indices, random_seed=42, verbose=False)

        rmse_scores = []

        for train_index, val_index in kf.split(X):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            train_pool = Pool(data=X_train, label=y_train, cat_features=cat_features_indices)
            val_pool = Pool(data=X_val, label=y_val, cat_features=cat_features_indices)

            model.fit(train_pool, eval_set=val_pool)

            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            rmse_scores.append(rmse)

        print(f"Average RMSE on {n_splits}-fold CV: {np.mean(rmse_scores)}")


def create_model():
    # Internal ipython tools

    # No warnings about setting value on copy of slice
    pd.options.mode.chained_assignment = None
    # Display up to 300 columns of a dataframe
    pd.set_option('display.max_columns', None)
    pd.set_option('display.float_format', '{:.3f}'.format)

    # Suppress warnings
    warnings.filterwarnings("ignore")

    # Set font size and style for plots
    plt.rcParams["font.size"] = 24
    sns.set(style="darkgrid", font_scale=2)
    sns.set_style({"font.family": "serif"})

    DEFAULT_RANDOM_SEED = 2023

    def set_seed_everything(seed: int = DEFAULT_RANDOM_SEED):
        """Set the random seeds for reproducibility"""
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        CatBoostRegressor.random_seed = seed
        print(f"Global seed set to {seed}")

    set_seed_everything(DEFAULT_RANDOM_SEED)

    # DATA_PATH = "/Prediction/app/Prediction/med.xlsx"
    DATA_PATH = "/Prediction/app/Prediction/med.xlsx"

    data = pd.read_excel(DATA_PATH).drop(columns=["Nr."])

    class MissingValuesAnalyzer:
        """
        Класс для анализа пропущенных значений в датафрейме.

        :param df: Датафрейм для анализа.
        :param mis_val_table: Таблица с информацией о пропусках в датафрейме.
        """

        def __init__(self, df: pd.DataFrame):
            self.df = df
            self.mis_val_table = None

        def _generate_missing_values_table(
                self, custom_missing_values: Optional[List[Any]] = None
        ) -> pd.DataFrame:
            """
            Расчет пропусков по столбцам в датафрейме.

            :param custom_missing_values: список дополнительных значений, которые считаются
                                          пропущенными (например, "None", "null", "missing", "unknown", "?").
            :return pd.DataFrame: таблица с информацией о пропусках в датафрейме.
            """
            if custom_missing_values is None:
                custom_missing_values = []

            # Находим стандартные пропуски (NaN) с использованием .isnull().sum()
            mis_val = self.df.isnull().sum()

            # Находим пропуски, представленные в виде специфических значений из списка custom_missing_values
            for value in custom_missing_values:
                mis_val += (self.df == value).sum()

            mis_val = mis_val[mis_val > 0]
            mis_val_percent = 100 * mis_val / len(self.df)

            self.mis_val_table = (
                pd.DataFrame(
                    {
                        "Missing Values": mis_val,
                        "% of Total Values": mis_val_percent,
                    }
                )
                .sort_values("% of Total Values", ascending=False)
                .round(1)
            )

        def _print_summary(self) -> None:
            """Выводит на экран информацию о количестве столбцов с пропущенными значениями."""
            if not self.mis_val_table.empty:
                print(
                    f"{self.mis_val_table.shape[0]}/{self.df.shape[1]} столбцов содержат пропущенные значения."
                )
            else:
                pass

        def _get_columns_to_remove(self, threshold_missing: float) -> List[str]:
            """Возвращает список столбцов c пропущенными значениями на основе порога threshold_missing."""
            missing_columns = list(
                self.mis_val_table[
                    self.mis_val_table["% of Total Values"] > threshold_missing
                    ].index
            )
            return missing_columns

        def find_columns_to_remove_by_threshold(
                self, threshold_missing: float
        ) -> List[str]:
            """
            Находит столбцы для удаления на основе заданного порога пропущенных значений.

            :param threshold_missing (float): Пороговое значение для удаления столбцов (в процентах).
            :return List[str]: Список столбцов, которые следует удалить.
            """
            self._generate_missing_values_table()
            # self._print_summary()
            return self._get_columns_to_remove(threshold_missing)

    class MissingValuesVisualizer:
        """
        Класс для визуализации пропущенных значений в датафрейме.

        :param df (pd.DataFrame): Датафрейм для анализа.
        :param mis_val_table (pd.DataFrame): Таблица с информацией о пропусках в датафрейме.
        """

        def __init__(self, df: pd.DataFrame, missing_values_table: pd.DataFrame):
            self.df = df
            self.mis_val_table = missing_values_table

        def plot_missing_bar(
                self,
                rotation: int = 0,
                alpha: float = 1.0,
                eps_x: float = 0.0,
                eps_y: float = 0.0,
                edgecolor: str = "white",
                figsize: tuple = (10, 15),
                color: str = "blue",
                title_size: int = 24,
                label_size: int = 12,
                ticks_size: int = 10,
                grid_linewidth: int = 2,
                title: str = "Распределение пропусков по признакам",
                title_pad: int = 10,
        ) -> None:
            """
            Визуализирует распределение пропусков по признакам.

            :param rotation (int, optional): Угол поворота подписей меток по оси X.
            :param alpha (float, optional): Прозрачность столбцов.
            :param eps_x (float, optional): Смещение по оси X для текста со значениями процентов пропусков.
            :param eps_y (float, optional): Смещение по оси Y для текста со значениями процентов пропусков.
            :param edgecolor (str, optional): Цвет краев столбцов.
            :param figsize (tuple, optional): Размер графика.
            :param color (str, optional): Цвет столбцов.
            :param title_size (int, optional): Размер шрифта заголовка.
            :param label_size (int, optional): Размер шрифта подписей процентов пропусков.
            :param ticks_size (int, optional): Размер шрифта меток осей.
            :param grid_linewidth (int, optional): Ширина линий сетки.
            :param title (str, optional): Заголовок графика.
            :param title_pad (int, optional): Отступ заголовка от верхней границы графика.
            """
            if self.mis_val_table.empty:
                print("Пропущенных значений не найдено.")
                return

            plt.figure(figsize=figsize)
            sns.barplot(
                y=self.mis_val_table.index,
                x=self.mis_val_table["Missing Values"],
                alpha=alpha,
                linewidth=1,
                edgecolor=edgecolor,
                color=color,
            )

            for i in range(len(self.mis_val_table)):
                plt.text(
                    y=i + eps_x,
                    x=self.mis_val_table["Missing Values"][i] + eps_y,
                    s=f"{self.mis_val_table['% of Total Values'][i]:.1f}",
                    color="black",
                    weight="bold",
                    va="center",
                    ha="center",
                    fontsize=label_size,
                )

            plt.title(title, fontsize=title_size, pad=title_pad)
            plt.xticks(rotation=rotation, fontsize=ticks_size)
            plt.yticks(fontsize=ticks_size)
            plt.grid(lw=grid_linewidth)
            plt.show()

    class MissingValuesImputer:
        """
        Класс для заполнения пропущенных значений в датафрейме.

        :field imputers (dict): Словарь с объектами Imputer для каждого столбца.
        """

        def __init__(self, df: pd.DataFrame):
            self.imputers = {}

        def fit_transform(
                self,
                X: pd.DataFrame,
                default_strategy: str = "mean",
                columns_strategy: Optional[Dict[str, str]] = None,
                categorical_fill_value: Optional[str] = "unknown",
        ) -> pd.DataFrame:
            """
            Заполнение пропусков в столбцах по отдельности нужной стратегией на тренировочном наборе.

            :param X: обучающий набор данных.
            :param columns_strategy: словарь, содержащий названия столбцов и соответствующие им стратегии заполнения.
                                     Возможные значения стратегий: 'mean', 'median', 'mode', 'categorical', 'knn'.
                                     По умолчанию None.
            :param default_strategy: cтратегия для всех остальных столбцов. По умолчанию 'mean'.
            :param categorical_fill_value: значение для заполнения категориальных переменных.
                                           Используется только в случае стратегии 'categorical'. По умолчанию 'unknown'.
            :return pd.DataFrame: новая таблица с заполненными пропусками для указанных столбцов.
            """
            filled_df = X.copy()

            if columns_strategy is None:
                columns_strategy = {}

            for col in X.columns:
                current_strategy = columns_strategy.get(col, default_strategy)

                if current_strategy in ["mean", "median", "most_frequent"]:
                    imputer = SimpleImputer(strategy=current_strategy)
                elif current_strategy == "categorical":
                    if not categorical_fill_value:
                        raise ValueError("categorical_fill_value должен быть указан.")
                    imputer = SimpleImputer(strategy="constant", fill_value=categorical_fill_value)

                filled_df[col] = imputer.fit_transform(filled_df[[col]]).ravel()
                self.imputers[col] = imputer

            return filled_df

        def transform(self, X: pd.DataFrame) -> pd.DataFrame:
            """
            Заполнение пропусков на валидационном и тестовом наборах.

            :param X: валидационный или тестовый набор данных.
            :return pd.DataFrame: новая таблица с заполненными пропусками для указанных столбцов.
            """
            filled_df = X.copy()

            for col, imputer in self.imputers.items():
                if col in filled_df.columns:
                    filled_df[col] = imputer.transform(filled_df[[col]]).ravel()

            return filled_df

    class FeatureTypeDetector:
        """
        Класс для определения типов признаков датафрейма, таких как категориальные, числовые,
        временные и текстовые признаки.

        :field df: Датафрейм для анализа.
        :field threshold_ratio: Пороговое значение для определения категориальных признаков.
        :field text_len_threshold: Пороговая длина текста для определения текстовых признаков.
        :field datetime_features: Кортеж названий временных признаков.
        :field categorical_features: Кортеж названий категориальных признаков.
        :field numerical_features: Кортеж названий числовых признаков.
        :field text_features: Кортеж названий текстовых признаков.
            """

        def __init__(
                self,
                df: pd.DataFrame,
                threshold_ratio: float = 0.1,
                text_len_threshold: int = 10,
        ):
            self.df = df
            self.threshold_ratio = threshold_ratio
            self.text_len_threshold = text_len_threshold

            self.detect_features()

        def detect_features(self) -> None:
            """Определяет типы признаков датафрейма."""
            # Определение временных признаков
            self.datetime_features = self.detect_datetime_features_by_df(self.df)

            # Определение категориальных признаков
            self.categorical_features_report_df = self.detect_categorical_features_by_df(
                self.df.drop(columns=self.datetime_features, errors="ignore"),
                threshold_ratio=self.threshold_ratio,
            )
            self.categorical_features = list(self.categorical_features_report_df.index)

            # Определение текстовых признаков
            self.text_features = self.detect_text_features_by_df(
                self.df.drop(columns=self.categorical_features, errors="ignore"),
                text_len_threshold=self.text_len_threshold,
            )

            # Определение числовых признаков
            all_features = set(self.df.columns)
            non_numerical_features = (
                    set(self.datetime_features)
                    | set(self.categorical_features)
                    | set(self.text_features)
            )
            self.numerical_features = list(all_features - non_numerical_features)

        @staticmethod
        def detect_datetime_features_by_df(df: pd.DataFrame) -> List[str]:
            """
            Ищет временные признаки среди столбцов датафрейма.

            :return List[str]: список названий временных признаков.
            """
            datetime_features = []

            for col in df.columns:
                # Проверяем только непропущенные значения
                datetime_col = df[col].dropna()

                if datetime_col.empty:
                    continue

                # Проверяем только столбцы с объектами
                if datetime_col.dtype.kind == "M":
                    datetime_features.append(col)
                elif datetime_col.dtype.kind == "O":
                    try:
                        # При помощи errors='coerce' преобразуем все непропущенные значения
                        # Значения, несоответствующие формату даты и времени, заменятся на NaT (Not-a-Time)
                        datetime_col = pd.to_datetime(df[col], errors="coerce")

                        if not datetime_col.isnull().all():  # Если все значения не являются NaT
                            datetime_features.append(col)
                    except Exception as e:
                        print(f"Error while checking column '{col}': {e}")

            return datetime_features

        @staticmethod
        def detect_categorical_feature(
                column: np.array,
                threshold_ratio: float = 0.1,
        ) -> Dict[str, Any]:
            """Определяет, является ли переданный столбец категориальным признаком."""
            # Проверка числа уникальных значений в столбце
            column = column.astype(str)  # Конвертируем столбец в строковый тип данных
            # .shape[0] вернёт кол-во элементов в этом массиве (число уникальных элементов)
            unique_count = np.unique(column, return_counts=False).shape[0]

            # Пропуск столбцов с числом уникальных значений больше порога
            if unique_count > len(column) * threshold_ratio:
                return None

            # Проверка типа данных
            dtype = column.dtype
            is_categorical = dtype == "object" or dtype.name.startswith("category")

            feature_info = {
                "feature_type": "str" if is_categorical else dtype,
                "unique_counts": unique_count,
                "value_counts": str(dict(zip(*np.unique(column, return_counts=True)))),
                "nan_ratios": round(pd.isnull(column).mean(), 2),
            }

            return feature_info

        @staticmethod
        def detect_categorical_features_by_df(
                df: pd.DataFrame,
                ignore_columns: Optional[List[str]] = None,
                threshold_ratio: float = 0.1,
        ) -> pd.DataFrame:
            """Ищет категориальные признаки среди столбцов датафрейма."""
            if ignore_columns is None:
                ignore_columns = []

            rows = {}

            for col in df.columns:
                if col in ignore_columns:
                    continue

                # Используем detect_categorical_feature для проверки каждого столбца
                feature_info = FeatureTypeDetector.detect_categorical_feature(
                    df[col].to_numpy(), threshold_ratio
                )

                if feature_info:
                    rows[col] = feature_info

            return pd.DataFrame(rows).T

        @staticmethod
        def detect_text_features_by_df(df: pd.DataFrame, text_len_threshold: int) -> List[str]:
            """
            Ищет текстовые признаки среди столбцов датафрейма.

            :param text_len_threshold: пороговая длина текста, при которой столбец считается текстовым признаком.
            :return List[str]: список названий текстовых признаков.
            """
            text_features = []

            for col in df.columns:
                # Проверяем только столбцы с объектами
                if df[col].dtype.kind == "O":
                    sample_value = df[col].dropna().iloc[0]
                    if isinstance(sample_value, str) and len(sample_value) > text_len_threshold:
                        text_features.append(col)

            return text_features

    class DataFrameMemoryOptimizer:
        """
        Класс для оптимизации использования памяти датафреймом
        преобразованием его столбцов в наиболее подходящие узкие типы данных.
        """

        def __init__(self):
            pass

        def memory_usage_report(func: Callable) -> Callable:
            """
            Декоратор для вывода отчета об использовании памяти датафреймом до и после оптимизации.

            :param func: функция, которую нужно обернуть декоратором.
            :return Callable: обернутая функция, которая выводит отчет об использовании памяти.
            """

            @wraps(func)
            def wrapper(*args, **kwargs):
                df = args[1]  # Используйте args[1] вместо args[0] для получения датафрейма
                start_mem = df.memory_usage().sum() / 1024 ** 2
                print(f"Memory usage of dataframe is {start_mem:.2f} MB")

                result = func(*args, **kwargs)

                end_mem = result.memory_usage().sum() / 1024 ** 2
                print(f"Memory usage after optimization is: {end_mem:.2f} MB")
                print(f"Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%")

                return result

            return wrapper

        @staticmethod
        def _get_optimal_numeric_type(
                dtype: str, col_min: float, col_max: float
        ) -> Optional[str]:
            """
            Возвращает наиболее подходящий числовой тип данных для столбца.

            :param dtype: Тип данных столбца ('int' или 'float').
            :param col_min: Минимальное значение столбца.
            :param col_max: Максимальное значение столбца.
            :return Optional[str]: Наиболее подходящий числовой тип данных или None,
                                    если dtype не является числовым.
            """
            numeric_types = {
                "int": [np.int8, np.int16, np.int32, np.int64],
                "float": [np.float16, np.float32, np.float64],
            }

            if dtype not in numeric_types:
                return None

            for t in numeric_types[dtype]:
                type_info = np.iinfo(t) if dtype == "int" else np.finfo(t)
                if col_min >= type_info.min and col_max <= type_info.max:
                    return t

            return None

        def _get_optimal_types(self, df: pd.DataFrame) -> Dict[str, str]:
            """
            Возвращает словарь с оптимальными типами данных для столбцов датафрейма.

            :param df: Датафрейм для анализа.
            :return Dict[str, str]: Словарь с оптимальными типами данных для столбцов.
            """
            types: Dict[str, str] = {}

            for col in df.columns:
                col_type = df[col].dtype

                # Если столбец имеет тип datetime64[ns], преобразуем его в datetime64[ms],
                # если данные не потеряют точность
                if (
                        col_type == "datetime64[ns]"
                        and (df[col] == df[col].astype("datetime64[ms]")).all()
                ):
                    types[col] = "datetime64[ms]"
                elif (
                        col_type == object and
                        FeatureTypeDetector.detect_categorical_feature(df[col].to_numpy())
                ):
                    types[col] = "category"
                elif (
                        col_type.kind in ("i", "f")  # проверяем, является ли тип данных числовым
                ):
                    optimal_type = self._get_optimal_numeric_type(
                        dtype="int" if str(col_type)[:3] == "int" else "float",
                        col_min=df[col].min(),
                        col_max=df[col].max(),
                    )
                    if optimal_type:
                        types[col] = optimal_type

            return types

        @memory_usage_report
        def optimize_df(self, df: pd.DataFrame) -> pd.DataFrame:
            """
            Оптимизирует использование памяти датафрейма, преобразуя его столбцы
            в наиболее подходящие типы данных.

            :param df: Датафрейм для оптимизации.
            :return pd.DataFrame: Оптимизированный датафрейм с измененными типами данных столбцов.
            """
            types = self._get_optimal_types(df)
            df = df.astype(types)
            return df

    memory_optimizer = DataFrameMemoryOptimizer()
    data = memory_optimizer.optimize_df(data)

    class DataFrameSummary:
        """
        Класс для анализа и генерации сводного отчета о датафрейме, включая количество строк и столбцов,
        типы данных, объем занимаемой памяти и график пропущенных значений.

        :field df: Датафрейм для анализа.
        :field row_count: Количество строк датафрейма.
        :field column_count: Количество столбцов датафрейма.
        :field memory_usage: Объем занимаемой памяти датафреймом (в мегабайтах).
        :field missing_values_analyzer: Объект для анализа пропущенных значений.
        :field missing_values_table: Таблица с информацией о пропущенных значениях.
        :field _feature_type_detector: Объект для определения типов признаков.
        :field categorical_features: Список категориальных признаков.
        :field numerical_features: Список числовых признаков.
        :field datetime_features: Список признаков даты и времени.
        :field text_features: Список текстовых признаков.
        """

        def __init__(self, df: pd.DataFrame):
            self.df = df
            self.row_count = df.shape[0]
            self.column_count = df.shape[1]
            self.memory_usage = df.memory_usage(index=True, deep=True).sum() / (1024 * 1024)

            # Анализ пропущенных значений
            self.missing_values_analyzer = MissingValuesAnalyzer(df)
            self.missing_values_analyzer._generate_missing_values_table()
            self.missing_values_table = self.missing_values_analyzer.mis_val_table

            # Определение типов признаков
            self._feature_type_detector = FeatureTypeDetector(df)
            self.categorical_features = self._feature_type_detector.categorical_features
            self.numerical_features = self._feature_type_detector.numerical_features
            self.datetime_features = self._feature_type_detector.datetime_features
            self.text_features = self._feature_type_detector.text_features

        def plot_missing_values(self):
            """Построение графика пропущенных значений."""
            missing_values_visualizer = MissingValuesVisualizer(
                self.df, self.missing_values_table
            )
            missing_values_visualizer.plot_missing_bar(
                ticks_size=16, label_size=14, grid_linewidth=3, figsize=(10, 6)
            )

        def generate_summary(self, plot_missing_values: bool = True) -> str:
            """
            Генерирует сводный отчет о датафрейме: количество строк и столбцов,
            типы данных, объем занимаемой памяти и график пропущенных значений.

            :param plot_missing_values: флаг, указывающий, нужно ли строить график пропущенных значений.
            :return str: строка с отчетом.
            """
            # if plot_missing_values:
            #     self.plot_missing_values()

            # Создаем словарь для подсчета типов данных
            dtype_counts = defaultdict(int)

            for dtype in self.df.dtypes:
                dtype_counts[str(dtype)] += 1

            dtype_report = "\n".join([
                f"{count} столбцов с типом {dtype}"
                for dtype, count in dtype_counts.items()
            ])
            dtype_report += "\n"

            categorical_features_str = "\n".join(self.categorical_features)
            numerical_features_str = "\n".join(self.numerical_features)
            datetime_features_str = "\n".join(self.datetime_features)
            text_features_str = "\n".join(self.text_features)

            feature_types_report = "\n".join([
                f"Категориальные признаки:\n{categorical_features_str}\n",
                f"Числовые признаки:\n{numerical_features_str}\n",
                f"Признаки даты и времени:\n{datetime_features_str}\n",
                f"Текстовые признаки:\n{text_features_str}\n",
            ])

            summary = "\n".join([
                f"Выводы:\n",
                f"в датафрейме {self.row_count} строк, {self.column_count} столбцов\n",
                dtype_report,
                feature_types_report,
                f"объем датафрейма {self.memory_usage:.1f}+ MB",
            ])

            return summary

    TARGET_COLUMN = "Total Cholesterol"

    data.dropna(subset=[TARGET_COLUMN], inplace=True)

    X = data.drop(TARGET_COLUMN, axis=1)
    y = data[TARGET_COLUMN]

    data_summary = DataFrameSummary(X)
    summary = data_summary.generate_summary()

    def remove_columns_by_name_and_dtype(
            df: pd.DataFrame,
            column_names: Optional[Iterable[str]] = None,
            dtypes: Optional[Iterable[Union[str, type]]] = None
    ) -> pd.DataFrame:
        """
        Удаляет столбцы из датафрейма по именам из списка или по типу данных.

        Args:
            df: исходный датафрейм.
            column_names: список столбцов для удаления.
            dtypes: список типов данных столбцов для удаления.

        Returns:
            pd.DataFrame: новый датафрейм с удаленными столбцами.
        """

        if column_names is None:
            column_names = set()

        if dtypes is None:
            dtypes = set()

        columns_to_remove = []

        for col in df.columns:
            if col in column_names or df[col].dtype in dtypes:
                columns_to_remove.append(col)

        return df.drop(columns=columns_to_remove)

    # Нахождение столбцов с пропущенными значениями по порогу
    missing_values_analyzer = MissingValuesAnalyzer(data)

    # Задаем порог для удаления колонок с пропущенными значениями
    threshold_missing = 30
    columns_to_remove = missing_values_analyzer.find_columns_to_remove_by_threshold(
        threshold_missing
    )

    print(f"Мы удалим {len(columns_to_remove)} столбцов (порог >= {threshold_missing}%):")

    def pretty_print_dict(d: Dict[Any, Any]) -> None:
        for key, value in d.items():
            print(f"{key} : {value}")

    pretty_print_dict(
        missing_values_analyzer.mis_val_table.loc[
            columns_to_remove, "% of Total Values"
        ].to_dict()
    )

    data = remove_columns_by_name_and_dtype(
        data,
        column_names=([
            "Annual Checkup CVRM (Russian version), Date Last",
            "Principal Practitioner Name CVRM",
            "Reference Date",
            "Date Last Interim Checkup CVRM (Dutch version)",
            "Annual Checkup CVRM (Dutch version), Date Last",
            "Date Last Interim Checkup CVRM (Russian version)",
            "MDRD",
            "MDRD, last measurement date",
            "Risk Score CVRM"
        ]),
    )

    # Оставшиеся столбцы с пропусками
    missing_values_analyzer.mis_val_table.drop(
        columns_to_remove, errors="ignore", inplace=True
    )

    # Делим на train и test.
    # X_train, X_test, y_train, y_test, scores_train, scores_test = train_test_split(
    #     data_pd, target_np, scores_np, test_size=0.2,
    #     stratify=target_np, random_state=RANDOM_STATE,
    # )
    #
    # # Делим на train и val.
    # X_train, X_val, y_train, y_val, scores_train, scores_val = train_test_split(
    #     X_train, y_train, scores_train, test_size=0.2,
    #     stratify=y_train, random_state=RANDOM_STATE,
    # )
    #
    # print_sample_sizes(X_train, y_train, X_val, y_val, X_test, y_test)
    #
    # # Заполнение пропущенных значений с использованием MissingValuesImputer
    # columns_strategy = {
    #     "ПО №3 Зеленод. ЦРБ": "categorical",
    # }
    #
    #
    # imputer = MissingValuesImputer(train_data)
    # train_data_filled = imputer.fit_transform(
    #     train_data, default_strategy="mean", columns_strategy=columns_strategy
    # )
    # val_data_filled = imputer.transform(val_data)
    # test_data_filled = imputer.transform(test_data)

    # Заполняю пропуски
    data['Hypertension'] = data['Hypertension'].cat.add_categories(['NaN'])
    data['Hypertension'].fillna('NaN', inplace=True)

    data['Smoking Status'] = data['Smoking Status'].cat.add_categories(['NaN'])
    data['Smoking Status'].fillna('NaN', inplace=True)

    categorical_features = ['Primary / Secondary CVRM', 'Hypertension',
                            'Patient Gender', 'Smoking Status',
                            'Organisation Name (CVRM Treatment)',
                            'Organisation Name (CVRM Treatment).1']

    numerical_features = ['Glucose Fasting', 'Systolic Blood Pressure',
                          'Diastolic Blood Pressure',
                          'BMI', 'Age']

    target_name = 'Total Cholesterol'

    date_features = ['Treatment Startdate CVRM',
                     'Glucose Fasting, last measurement date',
                     'Systolic Blood Pressure, last measurement date',
                     'BMI, last measurement date',
                     'Date Last Contact CVRM',
                     'Total Cholesterol, last measurement date']

    # Инициализируйте объект пайплайна
    pipeline = CatBoostPipeline(categorical_features=categorical_features,
                                numerical_features=numerical_features,
                                date_features=date_features)
    # Предобработка данных
    X, y = pipeline.preprocess_data(data=data, target_column=target_name)

    # # Подбор гиперпараметров
    # best_params, best_rmse = pipeline.hyperparameter_tuning(X, y)
    # print(f"Best params: {best_params}")
    # print(f"Best RMSE: {best_rmse}")

    # Запуск кросс-валидации
    pipeline.run_cross_validation(X, y)
