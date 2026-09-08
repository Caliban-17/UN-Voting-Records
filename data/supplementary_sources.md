# Supplementary sources for "Through which lens?"

Built by `scripts/build_supplementary.py` from the downloads below; the
outputs are keyed by the ISO-3 codes the voting record uses, with UN
lineages applied (the USSR takes Russia's series, Czechoslovakia Czechia's,
Yugoslavia and Serbia-Montenegro Serbia's; East and West Germany keep their
own).

| File | Content | Source | Licence |
|---|---|---|---|
| `world_bank_income_groups.csv` | income group per country and fiscal year, 1987–2023 | World Bank, "Historical classification by income" (OGHIST) | CC BY 4.0 |
| `vdem_regimes.csv` | Regimes of the World category (0 closed autocracy … 3 liberal democracy) and the liberal democracy index, 1946–2025 | V-Dem v15 (Coppedge et al.; Lührmann, Tannenberg and Lindberg 2018 for RoW) as processed and published by Our World in Data (`political-regime`, `liberal-democracy-index`) | CC BY 4.0 (OWID); V-Dem data free for research with citation |
| `women_in_parliament.csv` | share of seats held by women, 1946–2025, with `source` = `ipu` (1997 on) or `vdem` | Inter-Parliamentary Union via the World Bank indicator SG.GEN.PARL.ZS; V-Dem `v2lgfemleg` via Our World in Data (`share-of-women-in-parliament`) | CC BY 4.0 |
| `cow_defense_communities.csv` | per year to 2012, the modularity (Louvain) community of the defence-pact graph each state belongs to | Correlates of War Formal Alliances v4.1 (Gibler 2009), dyad-year file, `defense == 1`; state list `states2016.csv` for the code crosswalk | free for research with citation |

Citations:

- Coppedge, Michael, et al. 2025. "V-Dem Dataset v15." Varieties of Democracy Project.
- Lührmann, Anna, Marcus Tannenberg and Staffan Lindberg. 2018. "Regimes of the World (RoW): Opening New Avenues for the Comparative Study of Political Regimes." *Politics and Governance* 6(1).
- Gibler, Douglas M. 2009. *International Military Alliances, 1648–2008.* CQ Press. (Correlates of War Formal Alliances v4.1.)
- Inter-Parliamentary Union, "Women in national parliaments", via World Bank World Development Indicators.
- Our World in Data, processed series with the slugs named above.

The COW site refuses non-browser downloads; the copy used was the unmodified
dyad-year CSV published in the `joshloyal/dynetlsm` repository (identical
schema and row count to v4.1).

| `colonial_history.csv` | each state's former colonial ruler (ISO-3), independence year and ICOW independence type (1 formation, 2 decolonisation, 3 secession, 4 partition), and whether the ruler was an overseas empire | ICOW Colonial History Data Set v1.1 (Hensel 2018), `coldata110.csv`, via the COW state list for the code crosswalk | free for research with citation |
| `consensus_by_session.csv` | per session 74–80 and theme, the number of resolutions and the number adopted without a vote | DGACM extracts, `UNxml/GAresolutions` `data_extract/<session>.json`, field `adoption_type` | UN, for informational purposes |

- Hensel, Paul R. 2018. "ICOW Colonial History Data Set, version 1.1." http://www.paulhensel.org/icowcol.html
