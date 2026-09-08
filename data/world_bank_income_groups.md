# World Bank historical income classification

`world_bank_income_groups.csv` is a long-format extract of the World Bank's
"Historical classification by income" workbook (OGHIST.xlsx, sheet
"Country Analytical History"): one row per country and calendar year from
1987, with the analytical group `L` (low), `LM` (lower-middle), `UM`
(upper-middle) or `H` (high). Source: World Bank Country and Lending
Groups, https://datahelpdesk.worldbank.org/knowledgebase/articles/906519 —
licensed CC BY 4.0. Used by `src/lenses_partitions.py` to place members in
world-system tiers (core = H, semi-periphery = UM, periphery = LM/L).
Re-generate with the snippet in `scripts/` if the World Bank publishes a
new edition.
