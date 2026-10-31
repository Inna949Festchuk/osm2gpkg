#!/usr/bin/env python3
"""
OSM Data Pipeline для подготовки векторных масок для ML
Скачивает данные из Geofabrik, фильтрует и экспортирует в GeoPackage
"""

import os
import sys
import typer
import subprocess
import requests
from pathlib import Path
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import tempfile

from rich.console import Console
from rich.progress import (
    Progress, BarColumn, TextColumn, TimeRemainingColumn,
    TransferSpeedColumn, DownloadColumn, TaskID
)
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

# Геопроцессинг импорты (ленивые, чтобы CLI грузился быстрее)
import geopandas as gpd
from shapely.geometry import box
import pyproj

app = typer.Typer(
    name="osm-pipeline",
    help="🗺️  Pipeline для подготовки векторных данных OSM для ML",
    add_completion=False
)
console = Console()

# ==========================================
# КОНСТАНТЫ И КОНФИГУРАЦИЯ
# ==========================================

# Базовый URL Geofabrik
GEOFABRIK_BASE_URL = "https://download.geofabrik.de"

# Маппинг стран на поддиректории Geofabrik (континент/страна)
# Полный список: https://download.geofabrik.de/
COUNTRY_PATHS = {
    "russia": "europe/russia",
    "germany": "europe/germany",
    "france": "europe/france",
    "spain": "europe/spain",
    "italy": "europe/italy",
    "poland": "europe/poland",
    "ukraine": "europe/ukraine",
    "belarus": "europe/belarus",
    "kazakhstan": "asia/kazakhstan",
    "usa": "north-america/us",
    "canada": "north-america/canada",
    "brazil": "south-america/brazil",
    "china": "asia/china",
    "japan": "asia/japan",
    "india": "asia/india",
    "australia": "australia-oceania/australia",
    "united kingdom": "europe/united-kingdom",
    "netherlands": "europe/netherlands",
    "belgium": "europe/belgium",
    "sweden": "europe/sweden",
    "norway": "europe/norway",
    "finland": "europe/finland",
}

# Маппинг регионов (областей) внутри стран
# Формат: "название страны": {"название региона": "путь-в-geofabrik"}
REGION_PATHS = {
    "russia": {
        "moscow": "russia/moscow",
        "moskva": "russia/moscow",
        "saint petersburg": "russia/northwestern-fedistrict",
        "tula oblast": "russia/central-fedistrict",
        "tula": "russia/central-fedistrict",
        "moscow oblast": "russia/central-fedistrict",
        "leningrad oblast": "russia/northwestern-fedistrict",
        "crimea": "russia/crimean-fedistrict",
        "siberia": "russia/siberian-fedistrict",
        "ural": "russia/ural-fedistrict",
        "far east": "russia/far-eastern-fedistrict",
    },
    "germany": {
        "bavaria": "germany/bayern",
        "berlin": "germany/berlin",
        "hamburg": "germany/hamburg",
        "north rhine-westphalia": "germany/nordrhein-westfalen",
    },
    "usa": {
        "california": "us/california",
        "new york": "us/new-york",
        "texas": "us/texas",
        "florida": "us/florida",
    },
}

# Минимальные пороги фильтрации
MIN_AREA_M2 = 2500       # 0.25 га для лесов и водоемов
MIN_RIVER_WIDTH_M = 25   # Минимальная ширина русла для рек


# ==========================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================

def check_dependencies():
    """Проверяет наличие необходимых внешних утилит."""
    console.print("[bold]🔍 Проверка зависимостей...[/bold]")
    
    # Проверка osmium
    try:
        result = subprocess.run(
            ["osmium", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode != 0:
            console.print("[red]❌ osmium не установлен[/red]")
            console.print("   Установите: apt install osmium-tool (Linux)")
            console.print("   Или: brew install osmium-tool (macOS)")
            sys.exit(1)
        console.print("[green]✓ osmium доступен[/green]")
    except FileNotFoundError:
        console.print("[red]❌ osmium не найден в PATH[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]❌ Ошибка проверки osmium: {e}[/red]")
        sys.exit(1)


def resolve_geofabrik_url(country: str, region: Optional[str] = None) -> str:
    """Возвращает URL для скачивания .osm.pbf файла."""
    country_lower = country.lower().strip()
    
    if country_lower not in COUNTRY_PATHS:
        console.print(f"[yellow]⚠ Страна '{country}' не найдена в словаре[/yellow]")
        console.print(f"[yellow]  Доступные страны: {list(COUNTRY_PATHS.keys())}[/yellow]")
        # Пытаемся сформировать URL по умолчанию
        country_slug = country_lower.replace(" ", "-")
        path = f"europe/{country_slug}"
    else:
        path = COUNTRY_PATHS[country_lower]
    
    if region:
        region_lower = region.lower().strip()
        country_regions = REGION_PATHS.get(country_lower, {})
        
        if region_lower in country_regions:
            path = country_regions[region_lower]
        else:
            console.print(
                f"[yellow]⚠ Регион '{region}' для '{country}' не найден, "
                f"используется вся страна[/yellow]"
            )
    
    return f"{GEOFABRIK_BASE_URL}/{path}-latest.osm.pbf"


def download_file(
    url: str,
    save_path: Path,
    progress: Progress,
    task_id: TaskID
) -> Path:
    """
    Синхронная загрузка файла с прогресс-баром.
    Запускается в отдельном потоке ThreadPoolExecutor.
    """
    try:
        # Запрос с HEAD для получения размера файла
        head_response = requests.head(url, allow_redirects=True, timeout=30)
        head_response.raise_for_status()
        
        total_size = int(head_response.headers.get('content-length', 0))
        progress.update(task_id, total=total_size)
        progress.update(task_id, description=f"⬇️  {save_path.name}")
        
        # Скачивание с чанками
        downloaded = 0
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        progress.update(task_id, completed=downloaded)
        
        progress.update(task_id, description=f"✓ {save_path.name}")
        return save_path
        
    except requests.exceptions.RequestException as e:
        progress.update(task_id, description=f"❌ Ошибка: {save_path.name}")
        raise RuntimeError(f"Ошибка скачивания {url}: {e}")


def download_all_files(urls: List[str], output_dir: Path) -> List[Path]:
    """
    Параллельное скачивание нескольких файлов.
    Использует ThreadPoolExecutor для I/O-bound операций.
    """
    downloaded_files = []
    max_workers = min(len(urls), 4)  # Ограничиваем число потоков
    
    console.print(f"\n[bold yellow]📥 Скачивание {len(urls)} файл(ов)...[/bold yellow]")
    
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
        expand=True
    ) as progress:
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Создаем задачи
            futures = {}
            for url in urls:
                filename = url.split("/")[-1]
                save_path = output_dir / filename
                
                if save_path.exists():
                    console.print(f"[cyan]⏭  Файл уже существует: {filename}[/cyan]")
                    downloaded_files.append(save_path)
                    continue
                
                task_id = progress.add_task(f"⏳ {filename}", total=None)
                future = executor.submit(download_file, url, save_path, progress, task_id)
                futures[future] = save_path
            
            # Ожидаем завершения
            for future in as_completed(futures):
                try:
                    result = future.result()
                    downloaded_files.append(result)
                except Exception as e:
                    console.print(f"[red]❌ Ошибка загрузки: {e}[/red]")
                    sys.exit(1)
    
    return downloaded_files


def run_osmium_command(cmd: List[str], description: str = "") -> bool:
    """
    Запуск команды osmium через subprocess.
    """
    if description:
        console.print(f"[cyan]⚙️  {description}[/cyan]")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 час таймаут
            check=False
        )
        
        if result.returncode != 0:
            console.print(f"[red]❌ Ошибка osmium:[/red]")
            console.print(result.stderr)
            return False
        
        return True
        
    except subprocess.TimeoutExpired:
        console.print("[red]❌ Таймаут выполнения osmium[/red]")
        return False
    except Exception as e:
        console.print(f"[red]❌ Ошибка запуска osmium: {e}[/red]")
        return False


def merge_pbf_files(input_files: List[Path], output_file: Path) -> Path:
    """
    Объединяет несколько PBF файлов в один через osmium merge.
    """
    if len(input_files) == 1:
        return input_files[0]
    
    console.print(f"\n[bold yellow]🔗 Объединение {len(input_files)} файлов...[/bold yellow]")
    
    cmd = ["osmium", "merge"] + [str(f) for f in input_files] + ["-o", str(output_file)]
    
    if not run_osmium_command(cmd, "Слияние PBF файлов"):
        raise RuntimeError("Не удалось объединить PBF файлы")
    
    return output_file


def filter_pbf_with_osmium(input_file: Path, output_file: Path) -> Path:
    """
    Предварительная фильтрация PBF через osmium tags-filter.
    Оставляет только нужные теги для ускорения дальнейшей обработки.
    """
    console.print(f"\n[bold yellow]🔍 Предварительная фильтрация через Osmium...[/bold yellow]")
    
    # Фильтруем по тегам леса, воды и административных границ
    cmd = [
        "osmium", "tags-filter",
        str(input_file),
        # Лес
        "w/landuse=forest",
        "w/natural=wood",
        "wr/landuse=forest",
        "wr/natural=wood",
        # Вода
        "w/natural=water",
        "w/waterway=riverbank",
        "w/waterway=river",
        "w/waterway=canal",
        "w/waterway=stream",
        "wr/natural=water",
        # Города
        "w/boundary=administrative",
        "wr/boundary=administrative",
        # Населенные пункты
        "n/place=city",
        "n/place=town",
        "n/place=village",
        "-o", str(output_file),
        "--overwrite"
    ]
    
    if not run_osmium_command(cmd, "Фильтрация тегов"):
        raise RuntimeError("Не удалось отфильтровать PBF файл")
    
    # Показать размер результирующего файла
    size_mb = output_file.stat().st_size / (1024 * 1024)
    console.print(f"[green]✓ Отфильтрованный файл: {size_mb:.1f} MB[/green]")
    
    return output_file


def extract_bbox(input_file: Path, output_file: Path, bbox: tuple) -> Path:
    """
    Вырезает область по bbox из PBF файла через osmium extract.
    bbox: (min_lon, min_lat, max_lon, max_lat)
    """
    console.print(f"\n[bold yellow]✂️  Вырезание области по BBOX...[/bold yellow]")
    
    min_lon, min_lat, max_lon, max_lat = bbox
    
    cmd = [
        "osmium", "extract",
        "-b", f"{min_lon},{min_lat},{max_lon},{max_lat}",
        str(input_file),
        "-o", str(output_file),
        "--overwrite"
    ]
    
    if not run_osmium_command(cmd, f"BBOX: {bbox}"):
        raise RuntimeError("Не удалось вырезать область по BBOX")
    
    return output_file


# ==========================================
# ОБРАБОТКА СЛОЕВ (CPU-BOUND, В ОТДЕЛЬНЫХ ПРОЦЕССАХ)
# ==========================================

def get_utm_crs(lon: float, lat: float) -> str:
    """Определяет подходящую UTM проекцию по координатам."""
    utm_zone = int((lon + 180) / 6) + 1
    hemisphere = 'north' if lat >= 0 else 'south'
    return f"+proj=utm +zone={utm_zone} +{hemisphere} +datum=WGS84"


def process_forest_layer(pbf_path: Path, output_path: Path) -> Optional[gpd.GeoDataFrame]:
    """
    Обработка слоя "Лес".
    Загружается в отдельном процессе через ProcessPoolExecutor.
    """
    try:
        from pyrosm import OSM
        
        console.print("[cyan]🌲 Обработка слоя: Лес[/cyan]")
        
        # Загрузка данных
        osm = OSM(str(pbf_path))
        
        # Получаем полигоны лесов
        forest_gdf = osm.get_natural("wood")
        forest_landuse = osm.get_landuse("forest")
        
        # Объединяем, если есть данные
        gdfs = [gdf for gdf in [forest_gdf, forest_landuse] if gdf is not None and len(gdf) > 0]
        
        if not gdfs:
            console.print("[yellow]⚠ Лес не найден[/yellow]")
            return None
        
        forest = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
        
        # Переводим в метрическую проекцию для расчета площадей
        centroid = forest.geometry.unary_union.centroid
        utm_crs = get_utm_crs(centroid.x, centroid.y)
        forest_utm = forest.to_crs(utm_crs)
        
        # Фильтрация по площади >= 2500 м²
        forest_utm['area_m2'] = forest_utm.geometry.area
        forest_filtered = forest_utm[forest_utm['area_m2'] >= MIN_AREA_M2].copy()
        
        # Возвращаем в WGS84
        forest_final = forest_filtered.to_crs("EPSG:4326")
        
        # Выбираем только нужные колонки
        columns = ['id', 'name', 'area_m2', 'geometry']
        available_cols = [c for c in columns if c in forest_final.columns]
        forest_final = forest_final[available_cols].reset_index(drop=True)
        
        console.print(f"[green]✓ Лес: {len(forest_final)} объектов[/green]")
        
        # Сохраняем во временный файл для передачи между процессами
        forest_final.to_file(output_path, driver="GPKG", layer="forest")
        return output_path
        
    except Exception as e:
        console.print(f"[red]❌ Ошибка обработки леса: {e}[/red]")
        import traceback
        traceback.print_exc()
        return None


def process_city_layer(pbf_path: Path, output_path: Path, admin_levels: List[int]) -> Optional[Path]:
    """
    Обработка слоя "Город" (административные границы).
    """
    try:
        from pyrosm import OSM
        
        console.print("[cyan]🏙️  Обработка слоя: Город[/cyan]")
        
        osm = OSM(str(pbf_path))
        
        # Получаем административные границы
        boundaries = osm.get_boundaries(admin_level=admin_levels)
        
        if boundaries is None or len(boundaries) == 0:
            console.print("[yellow]⚠ Границы городов не найдены[/yellow]")
            return None
        
        # Фильтруем только полигоны и мультиполигоны
        city = boundaries[boundaries.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy()
        
        if len(city) == 0:
            console.print("[yellow]⚠ Полигональные границы не найдены[/yellow]")
            return None
        
        # Выбираем колонки
        columns = ['id', 'name', 'admin_level', 'geometry']
        available_cols = [c for c in columns if c in city.columns]
        city_final = city[available_cols].reset_index(drop=True)
        
        console.print(f"[green]✓ Города: {len(city_final)} объектов[/green]")
        
        city_final.to_file(output_path, driver="GPKG", layer="city")
        return output_path
        
    except Exception as e:
        console.print(f"[red]❌ Ошибка обработки городов: {e}[/red]")
        import traceback
        traceback.print_exc()
        return None


def process_water_layer(pbf_path: Path, output_path: Path) -> Optional[Path]:
    """
    Обработка слоя "Вода".
    Объединяет полигональные водоемы (>= 2500 м²) и линейные реки (width >= 25м).
    """
    try:
        from pyrosm import OSM
        import pandas as pd
        from shapely.geometry import MultiPolygon
        
        console.print("[cyan]💧 Обработка слоя: Вода[/cyan]")
        
        osm = OSM(str(pbf_path))
        
        # 1. Получаем полигональную воду
        water_polygons = osm.get_natural("water")
        riverbanks = osm.get_natural("riverbank")  # Иногда реки как полигоны
        
        poly_gdfs = []
        if water_polygons is not None and len(water_polygons) > 0:
            poly_gdfs.append(water_polygons)
        if riverbanks is not None and len(riverbanks) > 0:
            poly_gdfs.append(riverbanks)
        
        # 2. Получаем линейные объекты (реки, каналы)
        water_lines = None
        try:
            water_lines = osm.get_waterways()
        except:
            pass
        
        # Переводим в UTM для точных измерений
        all_geoms = []
        if poly_gdfs:
            all_geoms.extend(poly_gdfs)
        if water_lines is not None and len(water_lines) > 0:
            all_geoms.append(water_lines)
        
        if not all_geoms:
            console.print("[yellow]⚠ Вода не найдена[/yellow]")
            return None
        
        # Объединяем для определения центроида
        import pandas as pd
        temp_gdf = gpd.GeoDataFrame(pd.concat(all_geoms, ignore_index=True))
        centroid = temp_gdf.geometry.unary_union.centroid
        utm_crs = get_utm_crs(centroid.x, centroid.y)
        
        # ----- Обработка полигонов -----
        polygon_results = []
        for gdf in poly_gdfs:
            gdf_utm = gdf.to_crs(utm_crs)
            gdf_utm['area_m2'] = gdf_utm.geometry.area
            # Фильтр по площади
            filtered = gdf_utm[gdf_utm['area_m2'] >= MIN_AREA_M2].copy()
            if len(filtered) > 0:
                polygon_results.append(filtered)
        
        # ----- Обработка линий -----
        buffered_rivers = []
        if water_lines is not None and len(water_lines) > 0:
            lines_utm = water_lines.to_crs(utm_crs)
            
            # Проверяем наличие тега width
            if 'width' in lines_utm.columns:
                # Преобразуем width в числовой тип
                lines_utm['width_numeric'] = pd.to_numeric(
                    lines_utm['width'], errors='coerce'
                )
                
                # Фильтруем только линии с width >= 25м
                wide_rivers = lines_utm[
                    lines_utm['width_numeric'] >= MIN_RIVER_WIDTH_M
                ].copy()
                
                if len(wide_rivers) > 0:
                    # Буферизация: радиус = width / 2
                    wide_rivers['geometry'] = wide_rivers.apply(
                        lambda row: row.geometry.buffer(row.width_numeric / 2),
                        axis=1
                    )
                    
                    # Объединение с существующими полигонами для растворения границ
                    if polygon_results:
                        all_polys_utm = gpd.GeoDataFrame(
                            pd.concat(polygon_results, ignore_index=True),
                            crs=utm_crs
                        )
                        
                        # Удаляем линии, которые внутри полигонов (дубликаты)
                        # Проверяем пересечение с centroid линии
                        wide_rivers['centroid'] = wide_rivers.geometry.centroid
                        within_mask = gpd.sjoin(
                            wide_rivers[['centroid', 'geometry']],
                            all_polys_utm[['geometry']],
                            how='left',
                            predicate='within'
                        )
                        
                        # Оставляем только линии ВНЕ существующих полигонов
                        outside_mask = within_mask.index_right.isna()
                        wide_rivers = wide_rivers.loc[outside_mask.values]
                    
                    if len(wide_rivers) > 0:
                        buffered_rivers.append(wide_rivers[['geometry']])
        
        # ----- Финальное объединение -----
        all_water_parts = []
        
        # Полигоны
        if polygon_results:
            polys_utm = gpd.GeoDataFrame(
                pd.concat(polygon_results, ignore_index=True),
                crs=utm_crs
            )
            all_water_parts.append(polys_utm[['geometry']])
        
        # Буферизованные реки
        if buffered_rivers:
            rivers_gdf = gpd.GeoDataFrame(
                pd.concat(buffered_rivers, ignore_index=True),
                crs=utm_crs
            )
            all_water_parts.append(rivers_gdf)
        
        if not all_water_parts:
            console.print("[yellow]⚠ Вода после фильтрации не найдена[/yellow]")
            return None
        
        # Объединяем все в один GeoDataFrame
        water_combined = gpd.GeoDataFrame(
            pd.concat(all_water_parts, ignore_index=True),
            crs=utm_crs
        )
        
        # Растворение границ (dissolve) - объединяем пересекающиеся объекты
        water_dissolved = water_combined.dissolve()
        
        # Разбиваем мультиполигоны обратно на отдельные полигоны
        if len(water_dissolved) > 0:
            water_exploded = water_dissolved.explode(index_parts=False).reset_index(drop=True)
            
            # Финальная фильтрация по площади
            water_exploded['area_m2'] = water_exploded.geometry.area
            water_final = water_exploded[
                water_exploded['area_m2'] >= MIN_AREA_M2
            ].copy()
            
            # Возвращаем в WGS84
            water_final = water_final.to_crs("EPSG:4326")
            
            # Оставляем только нужные колонки
            water_final = water_final[['area_m2', 'geometry']].reset_index(drop=True)
            
            console.print(f"[green]✓ Вода: {len(water_final)} объектов[/green]")
            
            water_final.to_file(output_path, driver="GPKG", layer="water")
            return output_path
        
        console.print("[yellow]⚠ Итоговая вода пуста[/yellow]")
        return None
        
    except Exception as e:
        console.print(f"[red]❌ Ошибка обработки воды: {e}[/red]")
        import traceback
        traceback.print_exc()
        return None


# ==========================================
# ГЛАВНЫЙ PIPELINE
# ==========================================

def run_pipeline(
    countries: List[str],
    regions: Optional[List[str]],
    bbox: Optional[tuple],
    output: Path,
    workers: int,
    admin_levels: List[int],
    keep_temp: bool
):
    """
    Основной pipeline обработки OSM данных.
    """
    console.print(Panel.fit(
        "[bold blue]🗺️  OSM Data Pipeline для ML[/bold blue]\n"
        "Подготовка векторных масок из OpenStreetMap",
        border_style="blue"
    ))
    
    # Проверка зависимостей
    check_dependencies()
    
    # Создаем временную директорию для работы
    with tempfile.TemporaryDirectory(delete=not keep_temp) as temp_dir:
        temp_path = Path(temp_dir)
        
        # ========================================
        # ШАГ 1: Формирование списка URL для скачивания
        # ========================================
        console.print("\n[bold]📍 Шаг 1: Определение источников данных[/bold]")
        
        urls = []
        if countries:
            for country in countries:
                if regions:
                    for region in regions:
                        url = resolve_geofabrik_url(country, region)
                        urls.append(url)
                        console.print(f"  • {country} / {region}: {url}")
                else:
                    url = resolve_geofabrik_url(country)
                    urls.append(url)
                    console.print(f"  • {country}: {url}")
        else:
            # Если страны не указаны, используем дефолтную Россию
            url = resolve_geofabrik_url("russia")
            urls.append(url)
            console.print(f"  • По умолчанию (Россия): {url}")
        
        # ========================================
        # ШАГ 2: Скачивание файлов
        # ========================================
        console.print("\n[bold]📥 Шаг 2: Скачивание данных[/bold]")
        downloaded_files = download_all_files(urls, temp_path)
        
        if not downloaded_files:
            console.print("[red]❌ Не удалось скачать ни одного файла[/red]")
            sys.exit(1)
        
        # ========================================
        # ШАГ 3: Объединение файлов (если их несколько)
        # ========================================
        if len(downloaded_files) > 1:
            merged_pbf = temp_path / "merged.osm.pbf"
            current_pbf = merge_pbf_files(downloaded_files, merged_pbf)
        else:
            current_pbf = downloaded_files[0]
        
        # ========================================
        # ШАГ 4: Вырезание по BBOX (если указан)
        # ========================================
        if bbox:
            bbox_pbf = temp_path / "bbox_extract.osm.pbf"
            current_pbf = extract_bbox(current_pbf, bbox_pbf, bbox)
        
        # ========================================
        # ШАГ 5: Предварительная фильтрация через Osmium
        # ========================================
        filtered_pbf = temp_path / "filtered.osm.pbf"
        current_pbf = filter_pbf_with_osmium(current_pbf, filtered_pbf)
        
        # ========================================
        # ШАГ 6: Обработка слоев в параллельных процессах
        # ========================================
        console.print(f"\n[bold]🔧 Шаг 3: Обработка геометрии ({workers} ядер)[/bold]")
        
        # Пути для промежуточных результатов
        forest_path = temp_path / "forest.gpkg"
        water_path = temp_path / "water.gpkg"
        city_path = temp_path / "city.gpkg"
        
        # Запускаем обработку в пуле процессов
        with ProcessPoolExecutor(max_workers=workers) as pool:
            # Отправляем задачи параллельно
            future_forest = pool.submit(
                process_forest_layer,
                current_pbf,
                forest_path
            )
            
            future_water = pool.submit(
                process_water_layer,
                current_pbf,
                water_path
            )
            
            future_city = pool.submit(
                process_city_layer,
                current_pbf,
                city_path,
                admin_levels
            )
            
            # Ожидаем завершения всех задач
            results = {
                'forest': future_forest.result(),
                'water': future_water.result(),
                'city': future_city.result()
            }
        
        # ========================================
        # ШАГ 7: Финальное объединение в единый GeoPackage
        # ========================================
        console.print(f"\n[bold]💾 Шаг 4: Сохранение результата[/bold]")
        
        # Создаем итоговый GeoPackage
        output.parent.mkdir(parents=True, exist_ok=True)
        
        layers_saved = []
        
        for layer_name, layer_path in results.items():
            if layer_path and Path(layer_path).exists():
                try:
                    # Читаем слой
                    layer_gdf = gpd.read_file(layer_path, layer=layer_name)
                    
                    # Записываем в итоговый файл
                    mode = 'a' if output.exists() else 'w'
                    layer_gdf.to_file(
                        output,
                        driver="GPKG",
                        layer=layer_name,
                        mode=mode
                    )
                    layers_saved.append((layer_name, len(layer_gdf)))
                    console.print(f"[green]✓ Слой '{layer_name}' сохранен[/green]")
                    
                except Exception as e:
                    console.print(f"[red]❌ Ошибка сохранения {layer_name}: {e}[/red]")
        
        if not layers_saved:
            console.print("[red]❌ Не удалось сохранить ни одного слоя[/red]")
            sys.exit(1)
        
        # ========================================
        # ИТОГ: Вывод статистики
        # ========================================
        console.print()
        
        table = Table(title="📊 Результаты обработки", show_header=True)
        table.add_column("Слой", style="cyan", justify="left")
        table.add_column("Объектов", style="magenta", justify="right")
        table.add_column("Статус", style="green", justify="center")
        
        for layer_name, count in layers_saved:
            table.add_row(
                layer_name.capitalize(),
                f"{count:,}",
                "✓ Готов"
            )
        
        console.print(table)
        
        console.print(f"\n[bold green]✅ Файл сохранен: {output}[/bold green]")
        console.print(f"[dim]💡 Откройте в QGIS: qgis {output}[/dim]")
        
        # Временные файлы удалятся автоматически при выходе из with


# ==========================================
# CLI КОМАНДЫ
# ==========================================

@app.command()
def run(
    countries: List[str] = typer.Option(
        ["russia"],
        "--country", "-c",
        help="Названия стран (можно указывать несколько раз)"
    ),
    regions: Optional[List[str]] = typer.Option(
        None,
        "--region", "-r",
        help="Названия регионов/областей (можно указывать несколько раз)"
    ),
    bbox: Optional[str] = typer.Option(
        None,
        "--bbox", "-b",
        help="Bounding box в формате: min_lon,min_lat,max_lon,max_lat"
    ),
    output: Path = typer.Option(
        "./dataset.gpkg",
        "--output", "-o",
        help="Путь к выходному GeoPackage файлу"
    ),
    workers: int = typer.Option(
        4,
        "--workers", "-w",
        help="Количество ядер CPU для параллельной обработки"
    ),
    admin_levels: List[int] = typer.Option(
        [6, 8],
        "--admin-level",
        help="Уровни административных границ для городов (по умолчанию 6,8 для РФ)"
    ),
    keep_temp: bool = typer.Option(
        False,
        "--keep-temp",
        help="Не удалять временные файлы (для отладки)"
    ),
):
    """
    🗺️  Запуск pipeline для подготовки векторных данных OSM для ML.
    
    Примеры использования:
    
    \b
    # Скачать всю Россию
    python main.py run
    
    \b
    # Скачать несколько стран
    python main.py run --country Russia --country Germany
    
    \b
    # Скачать конкретные регионы
    python main.py run --country Russia --region "Tula Oblast" --region "Moscow"
    
    \b
    # Скачать по bbox
    python main.py run --country Russia --bbox "37.0,55.0,38.0,56.0"
    
    \b
    # Задать свой output и больше ядер
    python main.py run --country Russia --output ./my_data.gpkg --workers 8
    """
    # Парсинг bbox если передан
    bbox_tuple = None
    if bbox:
        try:
            parts = [float(x.strip()) for x in bbox.split(",")]
            if len(parts) != 4:
                raise ValueError
            bbox_tuple = tuple(parts)
            
            # Валидация координат
            min_lon, min_lat, max_lon, max_lat = bbox_tuple
            if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180):
                raise ValueError("Долгота должна быть в диапазоне [-180, 180]")
            if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90):
                raise ValueError("Широта должна быть в диапазоне [-90, 90]")
            if min_lon >= max_lon or min_lat >= max_lat:
                raise ValueError("Некорректный порядок координат")
                
        except ValueError as e:
            console.print(f"[red]❌ Некорректный формат bbox: {e}[/red]")
            console.print("[yellow]Формат: min_lon,min_lat,max_lon,max_lat[/yellow]")
            console.print("[yellow]Пример: 37.0,55.0,38.0,56.0[/yellow]")
            sys.exit(1)
    
    # Запуск pipeline
    try:
        run_pipeline(
            countries=countries,
            regions=regions,
            bbox=bbox_tuple,
            output=output,
            workers=workers,
            admin_levels=admin_levels,
            keep_temp=keep_temp
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Операция прервана пользователем[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]❌ Критическая ошибка: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


@app.command()
def list_countries():
    """
    📋 Показать список доступных стран и регионов.
    """
    console.print("\n[bold]🌍 Доступные страны:[/bold]\n")
    
    table = Table(show_header=True)
    table.add_column("Страна", style="cyan")
    table.add_column("Доступные регионы", style="magenta")
    
    for country in sorted(COUNTRY_PATHS.keys()):
        regions_list = REGION_PATHS.get(country, {})
        regions_str = ", ".join(sorted(regions_list.keys())) if regions_list else "-"
        table.add_row(country.capitalize(), regions_str)
    
    console.print(table)
    console.print("\n[dim]Используйте команду run с флагами --country и --region[/dim]")


@app.command()
def version():
    """🔢 Показать версию приложения."""
    console.print("[bold blue]OSM Data Pipeline[/bold blue]")
    console.print("Версия: [green]1.0.0[/green]")
    console.print("Автор: [cyan]OSM ML Team[/cyan]")


# ==========================================
# ТОЧКА ВХОДА
# ==========================================

if __name__ == "__main__":
    # Добавляем импорт pandas здесь, чтобы CLI загружался быстрее
    import pandas as pd
    app()