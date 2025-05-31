def wczytaj_ukladanke(nazwa_pliku):
    with open(nazwa_pliku, 'r') as plik:
        # Wczytanie pierwszej linii, która zawiera w i k
        w, k = map(int, plik.readline().strip().split())

        # Inicjalizacja pustej ramki o wymiarach w x k
        ukladanka = []

        # Wczytywanie kolejnych linii zawierających elementy układanki
        for linia in plik:
            elementy = list(map(int, linia.strip().split()))
            ukladanka.append(elementy)

    return ukladanka, w, k


def zapisz_rozwiazanie(nazwa_pliku, dlugosc_rozwiazania, ruchy):
    with open(nazwa_pliku, 'w') as plik:
        if dlugosc_rozwiazania == -1:
            # Jeśli nie znaleziono rozwiązania, zapisujemy tylko -1
            plik.write("-1\n")
        else:
            # W przeciwnym przypadku zapisujemy długość rozwiązania i ciąg ruchów
            plik.write(f"{dlugosc_rozwiazania}\n")
            plik.write(f"{ruchy}\n")


def zapisz_informacje_dodatkowe(nazwa_pliku, dlugosc_rozwiazania, liczba_stanow_odwiedzonych,
                                liczba_stanow_przetworzonych, max_glebokosc_rekursji, czas_trwania):
    with open(nazwa_pliku, 'w') as plik:
        plik.write(f"{dlugosc_rozwiazania}\n")
        plik.write(f"{liczba_stanow_odwiedzonych}\n")
        plik.write(f"{liczba_stanow_przetworzonych}\n")
        plik.write(f"{max_glebokosc_rekursji}\n")
        plik.write(f"{czas_trwania:.3f}\n")
