import argparse
import csv
import json
import os
import random
import re
import time
from datetime import date, timedelta
from typing import Dict, List, Optional
from urllib.parse import urlencode

from bs4 import BeautifulSoup
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

"""
python data_extraction/webscraping.py --step 1 --limit 5 --city Madrid --headed
python data_extraction/webscraping.py --step 2 --limit 100 --batch_size 2 --city Madrid --headed
"""

DEBUG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug")
URLS_FILE = "expedia_urls"


def build_search_url(check_in: str, check_out: str, city: str) -> str:
	params = {
		"destination": city,
		"adults": 2,
		"rooms": 1,
		"checkIn": check_in,
		"checkOut": check_out,
		"sort": "RECOMMENDED",
		"startDate": check_in,
		"endDate": check_out,
		"theme": "",
		"userIntent": "",
		"semdtl": "",
		"categorySearch": "",
		"useRewards": "false"
	}
	return f"https://www.expedia.es/Hotel-Search?{urlencode(params)}"


def setup_driver(headless: bool) -> uc.Chrome:
	options = uc.ChromeOptions()
	if headless:
		options.add_argument('--headless=new')
	options.add_argument('--disable-gpu')
	options.add_argument('--no-sandbox')
	options.add_argument('--disable-dev-shm-usage')
	options.add_argument('--disable-notifications')
	options.add_argument('--window-size=1920,1080')
	options.add_argument('--accept-lang=es-ES,es;q=0.9,en;q=0.8')

	driver = uc.Chrome(options=options, version_main=147)
	driver.implicitly_wait(10)
	return driver


def dismiss_popups(driver) -> None:
	popup_selectors = [
		"//button[contains(text(), 'Aceptar')]",
		"//button[contains(text(), 'Accept')]",
		"//button[contains(text(), 'Entendido')]",
		"//button[@aria-label='Cerrar']",
		"//button[@aria-label='Close']",
	]
	for selector in popup_selectors:
		try:
			btn = driver.find_elements(By.XPATH, selector)
			if btn and btn[0].is_displayed():
				btn[0].click()
				time.sleep(1)
		except Exception:
			pass


def extract_hotel_urls_from_search(driver, limit: int) -> List[str]:
	urls = set()
	current_page = 1
	max_pages = 20
	
	while len(urls) < limit and current_page <= max_pages:
		print(f"\nBuscando URLs en página {current_page} | URLs únicas encontradas: {len(urls)}")
		
		# Scroll to load dynamically
		scroll_height = driver.execute_script("return document.body.scrollHeight")
		for i in range(1, 4):
			target = (scroll_height * i) / 3
			driver.execute_script(f"window.scrollTo(0, {target});")
			time.sleep(random.uniform(1.0, 2.5))
		
		time.sleep(2)
		
		soup = BeautifulSoup(driver.page_source, "html.parser")
		link_elements = soup.select("a[data-stid='open-product-information']")
		
		for link in link_elements:
			if link.has_attr('href'):
				href = link['href']
				full_url = href if href.startswith("http") else f"https://www.expedia.es{href}"
				urls.add(full_url)
				if len(urls) >= limit:
					break
		
		if len(urls) >= limit:
			break
			
		try:
			next_btn_xpath = "//button[@data-stid='next-page'] | //a[contains(@aria-label, 'Siguiente')] | //button[contains(@aria-label, 'Siguiente')]"
			next_btns = driver.find_elements(By.XPATH, next_btn_xpath)
			
			if not next_btns:
				print("No hay botón de siguiente.")
				break
			
			next_btn = next_btns[0]
			if next_btn.get_attribute("aria-disabled") == "true" or not next_btn.is_enabled():
				break
			
			driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", next_btn)
			time.sleep(1)
			next_btn.click()
			time.sleep(random.uniform(4, 7))
		except Exception as e:
			print(f"Fin de paginación o error: {e}")
			break
			
		current_page += 1
		
	return list(urls)[:limit]


def extract_hotel_details(driver, url: str) -> Dict[str, Optional[str]]:
	print(f"  Visitando: {url.split('?')[0]}")
	driver.get(url)
	time.sleep(random.uniform(4, 7))
	dismiss_popups(driver)
	
	# Scroll a bit to load sections
	driver.execute_script("window.scrollBy(0, 1000);")
	time.sleep(1)
	
	soup = BeautifulSoup(driver.page_source, "html.parser")
	
	# Nombre
	name_elem = soup.select_one("h1.uitk-heading.uitk-heading-3")
	name = name_elem.get_text(strip=True) if name_elem else "Desconocido"
	
	# Dirección
	address_elem = soup.select_one("[data-stid='content-hotel-address']")
	address = address_elem.get_text(strip=True) if address_elem else None
	
	# Navegar y extraer valoración
	rating_elem = soup.select_one("span.uitk-badge-positive span.uitk-badge-base-text, span.uitk-badge-standard span.uitk-badge-base-text, div.uitk-text.uitk-type-700.uitk-type-bold.uitk-text-positive-theme")
	rating = rating_elem.get_text(strip=True) if rating_elem else "N/A"

	# Precio
	price = "N/A"
	price_elem = soup.select_one("div.range-indicator-badge span.uitk-badge-base-text, div[data-test-id='price-summary'] span, span.uitk-badge-base-text")
	if price_elem:
		price = price_elem.get_text(strip=True)
		
	# Si sigue capturando una puntuación (9,4), vamos a buscar por texto que contenga moneda
	if "€" not in price and "$" not in price:
		for p in soup.select("span.uitk-badge-base-text, div.uitk-text"):
			t = p.get_text(strip=True)
			if "€" in t or "$" in t:
				price = t
				break
				
	if price != "N/A":
		# Extraer solo los números
		numeric_price = re.sub(r'[^\d]', '', price)
		if numeric_price:
			price = numeric_price

	# Abrir el modal "Ver todo sobre este alojamiento"
	services = []
	try:
		# Botón para abrir modal
		btn_xpath = "//button[contains(@class, 'uitk-link') and contains(text(), 'Ver todo sobre este alojamiento')] | //button[@aria-label='Ver todo sobre este alojamiento']"
		modal_btn = WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.XPATH, btn_xpath)))
		driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", modal_btn)
		time.sleep(1)
		driver.execute_script("arguments[0].click();", modal_btn)
		
		# Esperar a que el modal se abra y cargue el tabpanel activo
		WebDriverWait(driver, 5).until(
			EC.presence_of_element_located((By.CSS_SELECTOR, "div[role='tabpanel'].uitk-tabs-pane.active"))
		)
		time.sleep(1) # Extra buffer para renderizado
		
		# Extraer servicios desde el nuevo HTML con el modal abierto
		modal_soup = BeautifulSoup(driver.page_source, "html.parser")
		active_tab = modal_soup.select_one("div[role='tabpanel'].uitk-tabs-pane.active")
		
		if active_tab:
			# Extrae todos los elementos de lista (típicamente los servicios concretos)
			list_items = active_tab.select("li.uitk-spacing, div.uitk-layout-grid div.uitk-text")
			for item in list_items:
				text = item.get_text(strip=True)
				if text and len(text) > 2 and text not in services:
					services.append(text)
		
	except Exception as e:
		print(f"     No se pudo abrir el modal o extraer del modal: {e}")
		# Fallback a lo que haya en la página principal
		service_blocks = soup.select("div.uitk-spacing.uitk-spacing-padding-blockstart-four span, [data-stid='hotel-amenities-list'] li")
		for s in service_blocks:
			text = s.get_text(strip=True)
			if text and text not in services:
				services.append(text)
			
	return {
		"name": name,
		"address": address,
		"rating": rating,
		"price": price,
		"services": services,
		"url": url
	}


def _parse_reviews_from_soup(soup, max_reviews: int) -> List[Dict[str, str]]:
	"""Extrae reviews del HTML parseado (score + comentario)."""
	reviews = []

	# Método 1: h3 con aria-label que contiene "de 10"
	review_score_elements = soup.select("h3.uitk-heading.uitk-heading-5[aria-label]")
	for score_elem in review_score_elements:
		if len(reviews) >= max_reviews:
			break

		aria_label = score_elem.get("aria-label", "")
		if "de 10" not in aria_label and "/10" not in score_elem.get_text(strip=True):
			continue

		raw_score = score_elem.get_text(strip=True)
		score_match = re.search(r'\d+', raw_score)
		score_text = score_match.group() if score_match else raw_score

		comment_text = ""
		parent = score_elem.parent
		for _ in range(8):
			if parent is None:
				break
			comment_divs = parent.select("div.uitk-text.uitk-type-300.uitk-text-standard-theme")
			if comment_divs:
				for cd in comment_divs:
					text = cd.get_text(strip=True)
					if text and len(text) > 10:
						comment_text = text
						break
			if comment_text:
				break
			parent = parent.parent

		if comment_text:
			reviews.append({"score": score_text, "comment": comment_text})

	# Método 2: contenedores de review con itemprop o data-stid
	if len(reviews) < max_reviews:
		review_containers = soup.select(
			"div[itemprop='review'], "
			"section[data-stid='reviews-section'] > div, "
			"div[data-stid='review-card']"
		)
		for container in review_containers:
			if len(reviews) >= max_reviews:
				break
			score_h3 = container.select_one("h3.uitk-heading")
			comment_div = container.select_one("div.uitk-text.uitk-type-300")
			if score_h3 and comment_div:
				raw_score = score_h3.get_text(strip=True)
				score_match = re.search(r'\d+', raw_score)
				score_text = score_match.group() if score_match else raw_score
				comment_text = comment_div.get_text(strip=True)
				if comment_text and len(comment_text) > 10:
					# Evitar duplicados
					if not any(r["comment"] == comment_text for r in reviews):
						reviews.append({"score": score_text, "comment": comment_text})

	# Método 3: buscar cualquier bloque con score tipo X/10 seguido de texto
	if len(reviews) < max_reviews:
		all_headings = soup.select("h3.uitk-heading")
		for h3 in all_headings:
			if len(reviews) >= max_reviews:
				break
			text = h3.get_text(strip=True)
			if re.match(r'\d+/10', text):
				raw_score = text
				score_match = re.search(r'\d+', raw_score)
				score_text = score_match.group() if score_match else raw_score
				# Buscar comentario cercano
				parent = h3.parent
				for _ in range(8):
					if parent is None:
						break
					comment_divs = parent.select("div.uitk-text.uitk-type-300")
					for cd in comment_divs:
						ct = cd.get_text(strip=True)
						if ct and len(ct) > 10 and not any(r["comment"] == ct for r in reviews):
							reviews.append({"score": score_text, "comment": ct})
							break
					if len(reviews) and reviews[-1].get("score") == score_text:
						break
					parent = parent.parent

	return reviews


def extract_hotel_reviews(driver, url: str, max_reviews: int = 5) -> List[Dict[str, str]]:
	"""Extrae reviews de un hotel de Expedia (score + comentario).
	Intenta primero extraer reviews visibles en la página, y si no hay
	suficientes, clicka el botón de comentarios y reintenta."""
	print(f"  Visitando para reviews: {url.split('?')[0]}")
	driver.get(url)
	time.sleep(random.uniform(4, 7))
	dismiss_popups(driver)

	# Scroll hacia abajo para cargar la sección de reviews
	for _ in range(5):
		driver.execute_script("window.scrollBy(0, 700);")
		time.sleep(random.uniform(0.8, 1.5))

	time.sleep(1)

	# Intento 1: extraer reviews ya visibles en la página
	soup = BeautifulSoup(driver.page_source, "html.parser")
	reviews = _parse_reviews_from_soup(soup, max_reviews)

	if len(reviews) >= max_reviews:
		for i, r in enumerate(reviews[:max_reviews]):
			print(f"    Review {i+1}: {r['score']} - {r['comment'][:60]}...")
		print(f"    Total reviews extraídas: {min(len(reviews), max_reviews)} (inline)")
		return reviews[:max_reviews]

	# Intento 2: clickar el botón "Abrir los XX comentarios"
	try:
		btn_xpath = (
			"//button[contains(@class, 'uitk-button-secondary') and contains(text(), 'comentarios')]"
			" | //button[contains(@class, 'uitk-button-secondary') and contains(text(), 'Abrir')]"
			" | //button[contains(@class, 'uitk-button') and contains(text(), 'comentarios')]"
		)
		review_btn = WebDriverWait(driver, 5).until(
			EC.element_to_be_clickable((By.XPATH, btn_xpath))
		)
		driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", review_btn)
		time.sleep(1)
		driver.execute_script("arguments[0].click();", review_btn)
		print("    Botón de comentarios clickado.")
		time.sleep(random.uniform(3, 5))

		# Re-parsear con el modal/sección abierta
		soup = BeautifulSoup(driver.page_source, "html.parser")
		reviews = _parse_reviews_from_soup(soup, max_reviews)
	except TimeoutException:
		print("    No se encontró botón de comentarios.")
	except Exception as e:
		print(f"    Error al clickar botón de comentarios: {e}")

	for i, r in enumerate(reviews[:max_reviews]):
		print(f"    Review {i+1}: {r['score']} - {r['comment'][:60]}...")
	print(f"    Total reviews extraídas: {min(len(reviews), max_reviews)}")
	return reviews[:max_reviews]


def fetch_reviews(headless: bool, batch_size: int) -> None:
	"""Extrae reviews de todos los hoteles y las guarda en data/expedia_hotels.json."""
	# Auto-detectar todas las ciudades disponibles en data/
	all_urls = []
	import glob
	url_files = sorted(glob.glob(f"data/{URLS_FILE}_*.json"))
	if not url_files:
		print("No se encontraron archivos de URLs en data/.")
		return
	for filepath in url_files:
		city = os.path.basename(filepath).replace(f"{URLS_FILE}_", "").replace(".json", "")
		with open(filepath, "r", encoding="utf-8") as f:
			urls = json.load(f)
			all_urls.extend(urls)
		print(f"Cargadas {len(urls)} URLs de {city}.")

	if not all_urls:
		print("No hay URLs para procesar.")
		return

	# Cargar hoteles existentes desde data/
	json_path = "data/expedia_hotels.json"
	all_hotels = []
	if os.path.exists(json_path):
		try:
			with open(json_path, "r", encoding="utf-8") as f:
				all_hotels = json.load(f)
		except Exception:
			pass

	# Crear índice de hoteles por URL para acceso rápido
	hotel_index = {}
	for i, h in enumerate(all_hotels):
		if "url" in h:
			hotel_index[h["url"]] = i

	# Solo procesar URLs que están en el dataset y NO tienen reviews aún
	skipped_with_reviews = 0
	skipped_not_found = 0
	seen = set()
	pending_urls = []
	for url in all_urls:
		if url in seen:
			continue
		seen.add(url)
		if url not in hotel_index:
			skipped_not_found += 1
			continue
		hotel = all_hotels[hotel_index[url]]
		if hotel.get("reviews"):
			skipped_with_reviews += 1
			continue
		pending_urls.append(url)

	print(f"  Saltados con reviews ya extraídas: {skipped_with_reviews}")
	print(f"  Saltados no encontrados en dataset: {skipped_not_found}")

	if not pending_urls:
		print("Todos los hoteles ya tienen reviews extraídas.")
		return

	urls_to_process = pending_urls[:batch_size]
	print(f"\n[PASO 3] Extrayendo reviews... (Pendientes: {len(pending_urls)}, En este lote: {len(urls_to_process)})")

	driver = None
	try:
		driver = setup_driver(headless)

		# Navegación inicial humana
		driver.get("https://www.google.com")
		time.sleep(random.uniform(2, 4))

		for i, url in enumerate(urls_to_process):
			print(f"\nProcesando hotel {i+1} de {len(urls_to_process)}...")
			try:
				reviews = extract_hotel_reviews(driver, url, max_reviews=5)

				if url in hotel_index:
					# Actualizar hotel existente
					all_hotels[hotel_index[url]]["reviews"] = reviews
				else:
					# Crear nueva entrada mínima
					all_hotels.append({
						"name": "Desconocido",
						"url": url,
						"reviews": reviews
					})
					hotel_index[url] = len(all_hotels) - 1

				# Checkpoint: guardar después de cada hotel
				with open(json_path, "w", encoding="utf-8") as f:
					json.dump(all_hotels, f, ensure_ascii=False, indent=2)
				print(f"    Checkpoint guardado.")

			except Exception as e:
				print(f"    Error extrayendo reviews: {e}")

			# Delay entre hoteles para no saturar
			time.sleep(random.uniform(2, 4))

	finally:
		if driver:
			try:
				driver.quit()
			except Exception:
				pass

	print(f"\nExtracción de reviews finalizada. Resultados guardados en {json_path}")


def fetch_urls(limit: int, headless: bool, city: str) -> List[str]:
	today = date.today()
	check_in = (today + timedelta(days=14)).isoformat()
	check_out = (today + timedelta(days=15)).isoformat()

	search_url = build_search_url(check_in, check_out, city)
	print(f"\n[PASO 1] URL de búsqueda: {search_url}")
	print(f"Buscando hasta {limit} URLs...")
	
	os.makedirs(DEBUG_DIR, exist_ok=True)
	driver = None
	urls = []
	
	try:
		driver = setup_driver(headless)
		
		# Navegación humana
		driver.get("https://www.google.com")
		time.sleep(random.uniform(2, 4))
		
		print("\nNavegando a la página de búsqueda...")
		driver.get(search_url)
		time.sleep(random.uniform(5, 8))
		dismiss_popups(driver)

		hotel_urls = extract_hotel_urls_from_search(driver, limit)
		print(f"\nSe encontraron {len(hotel_urls)} URLs de hoteles.")
		
		with open(f"{URLS_FILE}_{city}.json", "w", encoding="utf-8") as f:
			json.dump(hotel_urls, f, indent=2)
		print(f"URLs guardadas en {URLS_FILE}_{city}.json")
		
		return hotel_urls

	finally:
		if driver:
			try:
				driver.quit()
			except Exception:
				pass


def fetch_details(headless: bool, city: str, batch_size: int) -> List[Dict[str, Optional[str]]]:
	filename = f"{URLS_FILE}_{city}.json"
	if not os.path.exists(filename):
		print(f"No existe el archivo {filename}. Ejecuta primero el paso 1.")
		return []
		
	with open(filename, "r", encoding="utf-8") as f:
		hotel_urls = json.load(f)
		
	if not hotel_urls:
		print("La lista de URLs está vacía.")
		return []

	# Checkpoint: Leer los URLs que ya hemos procesado previamente
	processed_urls = set()
	if os.path.exists("expedia_hotels.json"):
		try:
			with open("expedia_hotels.json", "r", encoding="utf-8") as f:
				saved_data = json.load(f)
				for h in saved_data:
					if "url" in h:
						processed_urls.add(h["url"])
		except Exception:
			pass

	pending_urls = [u for u in hotel_urls if u not in processed_urls]
	
	if not pending_urls:
		print("Todos los hoteles de esta lista ya han sido procesados y guardados previamente.")
		return []

	print(f"\n[PASO 2] Procesando detalles... (Total base: {len(hotel_urls)}, Ya procesados: {len(processed_urls)}, Pendientes: {len(pending_urls)})")
	
	# Aplicar el tamaño por lote solo a los pendientes
	hotel_urls_to_process = pending_urls[:batch_size]
	print(f"-> Se extraerán {len(hotel_urls_to_process)} hoteles en este lote.")

	driver = None
	results = []
	
	try:
		driver = setup_driver(headless)
		
		for i, url in enumerate(hotel_urls_to_process):
			print(f"\nExtraendo hotel {i+1} de {len(hotel_urls_to_process)}...")
			try:
				details = extract_hotel_details(driver, url)
				details["city"] = city
				print(f"    Nombre: {details['name']}")
				print(f"    Ciudad: {details['city']}")
				print(f"    Rating: {details['rating']}")
				print(f"    Precio: {details['price']}")
				print(f"    Dirección: {details['address']}")
				print(f"    Servicios: {len(details['services'])} extraídos del modal")
				results.append(details)
			except Exception as e:
				print(f"    Error extrayendo detalles: {e}")
				
			# Checkpoint de seguridad inmediato: guardar en cada iteracion por si peta webdriver
			if len(results) > 0:
				save_results([results[-1]])
				
		return results

	finally:
		if driver:
			try:
				driver.quit()
			except Exception:
				pass


def save_results(hotels: List[Dict[str, Optional[str]]]) -> None:
	base_name = "expedia_hotels"
	json_path = f"{base_name}.json"
	csv_path = f"{base_name}.csv"

	all_hotels = []
	if os.path.exists(json_path):
		try:
			with open(json_path, "r", encoding="utf-8") as f:
				all_hotels = json.load(f)
		except Exception:
			pass
	all_hotels.extend(hotels)

	with open(json_path, "w", encoding="utf-8") as f:
		json.dump(all_hotels, f, ensure_ascii=False, indent=2)

	file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
	with open(csv_path, "a", newline="", encoding="utf-8") as f:
		if not hotels:
			return
		writer = csv.DictWriter(f, fieldnames=["city", "name", "price", "rating", "address", "services", "url"])
		if not file_exists:
			writer.writeheader()
		for h in hotels:
			h_csv = h.copy()
			h_csv["services"] = " | ".join(h["services"]) if h["services"] else ""
			writer.writerow(h_csv)


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Web scraping de hoteles en Expedia con undetected-chromedriver"
	)
	parser.add_argument(
		"--city",
		type=str,
		default="Barcelona (y alrededores), Cataluña, España",
		help="Ciudad a buscar (por defecto: Barcelona (y alrededores), Cataluña, España)",
	)
	parser.add_argument(
		"--limit",
		type=int,
		default=100,
		help="Numero maximo de hoteles a extraer (por defecto: 100)",
	)
	parser.add_argument(
		"--batch_size",
		type=int,
		default=100,
		help="Número máximo de hoteles a procesar por lote en el paso 2 o 3 (por defecto: 100)",
	)
	parser.add_argument(
		"--headed",
		action="store_true",
		help="Ejecuta el navegador en modo visible para depuracion",
	)
	parser.add_argument(
		"--step",
		choices=["1", "2", "3", "both"],
		default="both",
		help="Qué parte ejecutar: '1' URLs, '2' detalles, '3' reviews (Barcelona+Madrid). 'both' = pasos 1+2."
	)
	args = parser.parse_args()

	if args.step in ["1", "both"]:
		fetch_urls(limit=args.limit, headless=not args.headed, city=args.city)
		
	if args.step in ["2", "both"]:
		# "hotels" is returning all processed so far in the batch
		hotels = fetch_details(headless=not args.headed, city=args.city, batch_size=args.batch_size)
		if hotels:
			print(f"\nScraping por lotes finalizado. Hoteles extraídos en esta sesión: {len(hotels)}")
			print("Resultados añadidos a: expedia_hotels.json y expedia_hotels.csv")

	if args.step == "3":
		fetch_reviews(headless=not args.headed, batch_size=args.batch_size)


if __name__ == "__main__":
	main()
