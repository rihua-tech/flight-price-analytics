--1) Which routes are the most price-volatile?

--Why: volatile routes are good candidates for alerts and “Buy now” nudges.
SELECT
  route,
  COUNT(*)                AS days,
  ROUND(AVG(day_avg), 2)  AS avg_price,
  ROUND(STDDEV_SAMP(day_avg), 2) AS stdev
FROM (
  SELECT route, DATE(search_date) AS d, AVG(price) AS day_avg
  FROM public.fares_fact
  GROUP BY route, DATE(search_date)
) x
GROUP BY route
ORDER BY stdev DESC
LIMIT 10;

--2) What does the lead-time price curve look like for a route?

--Why: shows when booking is typically cheapest.
-- change route and date range to your needs
SELECT
  t.lead_band,
  COUNT(*)             AS n,
  ROUND(AVG(t.price),2) AS avg_price
FROM (
  SELECT price, lead_time_days,
         CASE
           WHEN lead_time_days <= 7  THEN '00–07'
           WHEN lead_time_days <= 14 THEN '08–14'
           WHEN lead_time_days <= 21 THEN '15–21'
           WHEN lead_time_days <= 28 THEN '22–28'
           WHEN lead_time_days <= 42 THEN '29–42'
           WHEN lead_time_days <= 60 THEN '43–60'
           ELSE '61+'
         END AS lead_band
  FROM public.fares_fact
  WHERE route = 'DXB-SIN'
    AND depart_date::date BETWEEN '2025-06-01' AND '2025-12-31'
) t
GROUP BY t.lead_band
ORDER BY MIN(t.lead_time_days);

--3) “Buy or Wait” for a given route & departure?

--Compare latest snapshot’s price to the last 30 days of snapshots (same route+depart_date). Flag if latest is ≥10% below the 30-day median.

-- Set your route & depart date here
WITH params AS (
  SELECT 'DXB-SIN'::text AS route, DATE '2025-09-15' AS depart
),
mx AS (
  SELECT MAX(f.snapshot_date) AS mx
  FROM public.fares_fact f, params p
  WHERE f.route = p.route
    AND f.depart_date::date = p.depart
),
latest AS (
  SELECT f.price, f.snapshot_date
  FROM public.fares_fact f, params p, mx
  WHERE f.route = p.route
    AND f.depart_date::date = p.depart
  ORDER BY f.snapshot_date DESC
  LIMIT 1
),
hist AS (
  SELECT
    -- cast to numeric so ROUND(..., 2) works
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY f.price)::numeric AS median_30d
  FROM public.fares_fact f, params p, mx
  WHERE f.route = p.route
    AND f.depart_date::date = p.depart
    AND f.snapshot_date >= mx.mx - INTERVAL '30 days'
    AND f.snapshot_date <  mx.mx
)
SELECT
  l.price                                   AS latest_price,
  ROUND(h.median_30d, 2)                    AS median_30d,
  CASE
    WHEN l.price <= 0.90 * h.median_30d THEN 'BUY'
    ELSE 'WAIT'
  END                                       AS recommendation
FROM latest l
CROSS JOIN hist h;


--4) What day of week is typically cheapest to depart (by route)?

--Why: easy content for users and marketing (“Tuesdays are cheaper on DXB-SIN”).

SELECT
  route,
  TO_CHAR(depart_date, 'Dy') AS depart_dow,
  ROUND(AVG(price), 2)       AS avg_price,
  COUNT(*)                   AS n
FROM public.fares_fact
GROUP BY route, depart_dow
ORDER BY route, depart_dow;


--5) What booking window (lead time) tends to have the lowest average price per route?

--Why: summarize “sweet spot” for each route.
-- For each route, pick the lead_time_days with the lowest avg price
SELECT DISTINCT ON (route)
  route,
  lead_time_days AS best_lead_time_days,
  ROUND(avg_price,2) AS best_avg_price,
  n
FROM (
  SELECT route, lead_time_days, AVG(price) AS avg_price, COUNT(*) AS n
  FROM public.fares_fact
  GROUP BY route, lead_time_days
) a
ORDER BY route, avg_price ASC, n DESC;



