# Corpus documental (RAG)

Descargado el **2026-11-06**. Los metadatos que usa la ingesta están en `sources.json`.

| Archivo | Documento | URL |
|---|---|---|
| `res_2254_2017_minambiente.pdf` | Resolución 2254 de 2017 (MinAmbiente), norma de calidad del aire. Copia escaneada con OCR publicada por la CVC; la copia de minambiente.gov.co es una imagen sin texto | https://calidadaire.cvc.gov.co/sites/default/files/2021-05/Resoluci%C3%B3n%202254%20de%202017.pdf |
| `acuerdo_metropolitano_15_poeca.pdf` | Acuerdo Metropolitano N.° 15, Protocolo POECA del AMVA (versión alojada por SICLAS) | https://www.siclas.org/wp-content/uploads/2017/02/AcuerdoProtocoloPOECA_FINAL.pdf |
| `resolucion_metropolitana_episodio_2022.pdf` | Resolución Metropolitana que declara el periodo de gestión de episodios, 2.º semestre de 2022 | https://www.corantioquia.gov.co/wp-content/uploads/2022/10/ResolucionDeclaracionEpisodio2doSemestre2022.pdf |
| `ideam_hoja_metodologica_ica_2025.pdf` | IDEAM, hoja metodológica del Índice de Calidad del Aire (v03, 2025) | https://bart.ideam.gov.co/indiecosistemas/ind/aire/hm/1.%20HM%20Indice%20calidad%20aire.pdf |
| `oms_directrices_calidad_aire_2021_resumen.pdf` | OMS, Directrices mundiales sobre la calidad del aire 2021, resumen ejecutivo (es) | https://iris.who.int/handle/10665/346062 |
| `ops_guias_oms_2021_presentacion.pdf` | OPS, presentación sobre las guías OMS 2021 y herramientas | https://paho.org/sites/default/files/ops_aqg_y_herramientas_de_evaluacion_0.pdf |
| `la_estrella_plan_accion_poeca_2019.pdf` | Municipio de La Estrella, plan de acción POECA 2019 (medidas municipales y recomendaciones) | https://2024.laestrella.gov.co/uploads/entidad/control/6f9eb-plan-de-accion-poeca.pdf |

**Pendiente / no conseguido:** el Acuerdo Metropolitano 04 de 2018 (POECA vigente) y su modificación de 2021.
La URL oficial en metropol.gov.co devolvió 404 y el portal de datos del AMVA respondió 502 el 2026-11-06.
Si el equipo lo consigue, se agrega aquí y en `sources.json` y se corre `python -m src.rag.ingest`.
