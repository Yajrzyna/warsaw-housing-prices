"""Code lists of the RCN conceptual model (zalacznik nr 9, EGiB regulation 2021)."""

RODZAJ_TRANSAKCJI = {
    1: "wolny_rynek",              # free market
    2: "sprzedaz_bezprzetargowa",  # non-tender sale (public seller)
    3: "sprzedaz_przetargowa",     # tender / auction sale
    4: "sprzedaz_egzekucyjna",     # enforcement (bailiff) sale
    5: "sprzedaz_na_cel_publiczny",
    6: "sprzedaz_z_bonifikata",    # discounted sale (typically municipal flat to tenant)
}

RODZAJ_RYNKU = {1: "pierwotny", 2: "wtorny"}  # primary / secondary

STRONA = {  # seller / buyer type
    1: "skarb_panstwa",
    2: "jst",            # local government
    3: "osoba_fizyczna",
    4: "osoba_prawna",
}

RODZAJ_NIERUCHOMOSCI = {
    1: "gruntowa_niezabudowana",
    2: "gruntowa_zabudowana",
    3: "budynkowa",
    4: "lokalowa",
}

RODZAJ_PRAWA = {
    1: "wlasnosc_gruntu",
    2: "uzytkowanie_wieczyste",
    3: "wlasnosc_lokalu",
    4: "wlasnosc_budynku",
}

FUNKCJA_LOKALU = {
    1: "mieszkalna",
    2: "handlowo_uslugowa",
    3: "biurowa",
    4: "produkcyjna",
    5: "garaz",
    6: "inna",
}

RODZAJ_BUDYNKU = {
    101: "przemyslowy", 102: "transportu_i_lacznosci", 103: "handlowo_uslugowy",
    104: "zbiorniki_silosy_magazyny", 105: "biurowy", 106: "szpitale",
    107: "oswiaty_i_sportu", 108: "gospodarczy", 109: "pozostale_niemieszkalne",
    110: "mieszkalny",
}

SPOSOB_UZYTKOWANIA = {
    1: "grunty_rolne", 2: "grunty_lesne", 3: "grunty_zabudowane_i_zurbanizowane",
    4: "tereny_komunikacyjne", 5: "inne",
}
