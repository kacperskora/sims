# Module 1: Coin Flip Betting & the Kelly Criterion — Theory

Ten dokument wyjaśnia matematykę stojącą za modułem 1 oraz to, jak konkretne
wzory przekładają się na kod w `src/quant_sims/betting/`.

---

## 1. Setup gry

Rozważamy powtarzany zakład o wynik rzutu (niekoniecznie uczciwą) monetą:

- **p** — prawdopodobieństwo wygranej (0 < p < 1)
- **b** — "net odds", czyli stosunek wypłaty do stawki. Jeśli obstawiasz
  stawkę *s* i wygrywasz, dostajesz *s + b·s* (czysty zysk = *b·s*). Jeśli
  przegrywasz, tracisz całą stawkę *s*. Przy b = 1.0 mamy zakład "1:1"
  (fair-odds coin — wygrana podwaja stawkę).
- **f** — ułamek bieżącego kapitału, jaki stawiamy w danej rundzie (0 ≤ f ≤ 1)

To bezpośrednio odpowiada klasie `CoinFlipGame(p_win, payout_ratio, seed)`
w `coin_flip.py`.

**Wartość oczekiwana zakładu** (edge) przy stawce jednostkowej:

```
E[X] = p·b − (1 − p) = p·b − q,    gdzie q = 1 − p
```

Gra ma dodatnią wartość oczekiwaną (positive edge) wtedy i tylko wtedy, gdy
`p·b > q`. Odpowiada temu funkcja `has_positive_edge(p, b)` w `kelly.py`.

---

## 2. Dlaczego nie stawiać zawsze 100%? — problem z EV

Naiwna intuicja mówi: "skoro edge jest dodatni, stawiaj tyle ile możesz".
To błąd. Przy stawianiu ułamka `f = 1.0` (all-in) każdego kapitału, **jedna
przegrana zeruje cały kapitał** — i to niezależnie od tego, jak wiele
wcześniejszych wygranych było. Ponieważ przy dowolnie długiej serii rzutów
prawdopodobieństwo co najmniej jednej przegranej dąży do 1, strategia all-in
prowadzi do bankructwa **prawie na pewno** (almost surely), nawet przy
silnym dodatnim edge'u. To testuje `test_all_in_can_reach_ruin` i
`test_ruin_probability_montecarlo_is_high_for_all_in`.

Z drugiej strony, stawianie zbyt małego ułamka (np. f → 0) jest "bezpieczne",
ale marnuje edge — kapitał rośnie wolniej niż mógłby.

Rodzi to naturalne pytanie: **jaki ułamek f maksymalizuje długoterminowe
tempo wzrostu kapitału?**

---

## 3. Kryterium Kelly'ego — wyprowadzenie

### 3.1 Dlaczego logarytm?

Przy powtarzanym stawianiu ułamka kapitału, kapitał po *n* rundach to
iloczyn (nie suma) czynników wzrostu:

```
C_n = C_0 · ∏(1 + f·b)^{wygrane} · (1 − f)^{przegrane}
```

Maksymalizacja *oczekiwanego kapitału* E[C_n] prowadzi właśnie do strategii
all-in (bo iloczyn wartości oczekiwanych jednego kroku jest maksymalizowany
przy f=1, mimo katastrofalnego ryzyka ruiny). To zły cel.

Zamiast tego Kelly maksymalizuje **oczekiwane tempo wzrostu logarytmu
kapitału** — co odpowiada maksymalizacji *mediany* (typowej trajektorii),
a nie średniej zdominowanej przez rzadkie, ekstremalne wygrane:

```
g(f) = E[log(C_n / C_0)] / n = p·log(1 + f·b) + q·log(1 − f)
```

To dokładnie funkcja `expected_log_growth(p, b, f)` w `kelly.py`.

### 3.2 Znajdowanie optimum

Różniczkujemy g(f) po f i przyrównujemy do zera:

```
g'(f) = p·b/(1 + f·b) − q/(1 − f) = 0
```

Rozwiązując:

```
p·b·(1 − f) = q·(1 + f·b)
p·b − p·b·f = q + q·b·f
p·b − q = f·b·(p + q)
p·b − q = f·b          (bo p + q = 1)

f* = (p·b − q) / b = p − q/b
```

To jest **wzór Kelly'ego**, zaimplementowany w `kelly_fraction(p, b)`:

```python
f_star = p - q / b
```

Druga pochodna g''(f) jest zawsze ujemna w dopuszczalnym zakresie
`(-1/b, 1)`, więc to jest maksimum globalne (funkcja jest wklęsła) —
stąd test `test_expected_log_growth_at_kelly_is_maximum`, który numerycznie
sprawdza, że lekkie odchylenie od f* w dowolną stronę zmniejsza g(f).

### 3.3 Interpretacja

- Gdy `p·b > q` (dodatni edge) → f* > 0 → warto grać
- Gdy `p·b = q` (fair game, brak edge'u) → f* = 0 → nie graj wcale (patrz
  `test_kelly_fraction_fair_coin_no_edge`, p=0.5, b=1.0 → f*=0)
- Gdy `p·b < q` (ujemny edge) → f* < 0 → optymalnie jest obstawiać
  przeciwną stronę zakładu; jeśli to niemożliwe, po prostu nie grać
  (`kelly_strategy` w kodzie clipuje ujemne f* do zera — patrz
  `test_kelly_strategy_clips_negative_edge_to_zero`)

**Przykład liczbowy** (klasyczny z podręczników): p=0.6, b=1.0 →
f* = 0.6 − 0.4/1.0 = **0.2**, czyli 20% kapitału na zakład. Weryfikuje to
`test_kelly_fraction_known_case`.

---

## 4. Strategie w kodzie — co która robi matematycznie

Wszystkie strategie w `kelly.py` mają wspólny interfejs:
`strategy(capital, history) -> f`, gdzie `history` to lista dotychczasowych
wyników (True = wygrana).

| Strategia | Wzór na f | Sens |
|---|---|---|
| `fixed_fraction(f)` | f = const | Baseline — stały ułamek niezależnie od wyników |
| `kelly_strategy(p, b, mult=1.0)` | f = mult · max(f*, 0) | Pełny Kelly przy mult=1, half-Kelly przy mult=0.5 |
| `martingale(base, max_f)` | f = min(base · 2^{streak przegranych}, max_f) | Podwajanie stawki po każdej przegranej — próba "odbicia się" jedną wygraną |
| `all_in()` | f = 1.0 zawsze | Maksymalny risk, testowy punkt odniesienia |

### Dlaczego half-Kelly ma sens praktyczny

Mnożąc f* przez stały współczynnik `mult ∈ (0,1)`, tracimy część tempa
wzrostu, ale **nieproporcjonalnie mniej redukujemy wariancję**. To dlatego,
że g(f) jest wklęsła i płaska blisko maksimum (pochodna bliska zeru), a
funkcja wariancji trajektorii rośnie znacznie szybciej z odejściem od
zachowawczych wartości f. W praktyce (trading, hazard) prawie nikt nie gra
pełnym Kelly z powodu:
1. niepewności co do prawdziwego p (błąd estymacji przekłada się na
   przeszacowanie f*)
2. akceptowalności obsunięć kapitału (drawdown) po drodze do celu

### Martingale — dlaczego to pułapka

Martingale nie zmienia wartości oczekiwanej pojedynczej rundy — to wciąż
gra o tym samym rozkładzie p, b. Zmienia tylko **kształt rozkładu wyniku**:
wysokie prawdopodobieństwo małej wygranej (seria się urywa na wygranej) i
niskie prawdopodobieństwo katastrofalnej straty (długa seria przegranych,
przy której stawka rośnie wykładniczo — 2^streak). Cap `max_fraction`
istnieje w kodzie właśnie po to, żeby symulacja nie próbowała postawić
np. 1600% kapitału po 5 przegranych z rzędu.

---

## 5. Symulacja pojedynczej serii — `simulate_series`

Metoda `CoinFlipGame.simulate_series` implementuje dokładnie proces opisany
w sekcji 3.1, krok po kroku:

```
dla każdej rundy i:
    f_i = strategy(capital, history)          # zapytaj strategię o ułamek
    stake = f_i · capital
    win = flip()                                # rzut monetą z prawdopodobieństwem p
    if win:  capital += stake · b               # wygrana: +b razy stawka
    else:    capital -= stake                    # przegrana: cała stawka
    if capital <= ruin_threshold: ruined = True   # bankructwo
```

`ruin_threshold` (domyślnie 1e-6, praktycznie zero) obsługuje błędy
zmiennoprzecinkowe przy kapitale asymptotycznie dążącym do zera. Po
osiągnięciu ruiny ścieżka jest "płaska" (capital pozostaje 0), żeby
wszystkie symulacje w batchu miały tę samą długość (potrzebne do
wektoryzacji w `monte_carlo`).

---

## 6. Monte Carlo — dlaczego symulacja, nie tylko wzór analityczny

Wzór na f* mówi, jaki ułamek jest *optymalny w oczekiwaniu*, ale nie mówi
nic o **rozkładzie** możliwych wyników po *n* rundach — a to jest kluczowe
dla realnej oceny ryzyka. `CoinFlipGame.monte_carlo` uruchamia
`simulate_series` N razy niezależnie i zwraca macierz ścieżek kapitału
`(n_simulations, n_flips+1)`. Na tej macierzy liczymy:

- **medianę ścieżek** — typowy wynik (to jest to, co Kelly faktycznie
  maksymalizuje)
- **rozkład kapitału końcowego** (histogram) — pokazuje asymetrię: nawet
  przy dobrej strategii część symulacji kończy się gorzej niż start
- **prawdopodobieństwo ruiny** — patrz sekcja 7

## 7. Prawdopodobieństwo ruiny — dlaczego bez wzoru analitycznego

Klasyczny **problem gracza (gambler's ruin)** ma piękne zamknięte wzory —
ale tylko dla stawek o **stałej wielkości bezwzględnej** (np. zawsze
1 zł), nie stałego *ułamka* kapitału. Przy stawianiu ułamka f każdej rundy,
kapitał matematycznie nigdy nie osiąga dokładnie zera przy skończonej
liczbie rund (to proces multiplikatywny — może tylko asymptotycznie zbliżać
się do zera). W praktyce (i w realnym tradingu) definiujemy "ruinę" jako
spadek poniżej pewnego progu (np. 5% kapitału początkowego) — i to jest coś,
co dużo naturalniej estymuje się empirycznie niż wyprowadza analitycznie.

Stąd `ruin_probability_montecarlo`:

```
ruin_prob = (liczba symulacji, w których kapitał spadł ≤ ruin_level) / N
```

a `ruin_probability_analytic` jest w kodzie celowo zaimplementowana jako
`NotImplementedError` z komentarzem wyjaśniającym dlaczego — to
udokumentowana decyzja projektowa, nie brak funkcjonalności.

---

## 8. Co pokazują wykresy w notebooku

- **`plot_growth_rate_curve`** — rysuje g(f) z sekcji 3.1 dla zakresu f,
  z pionową linią w f*. Wizualnie potwierdza, że f* to wierzchołek paraboli-
  podobnej, wklęsłej krzywej.
- **`plot_equity_curves`** — pojedyncze trajektorie kapitału (log-scale na
  osi Y, bo wzrost jest multiplikatywny/wykładniczy) plus mediana na tle
  wszystkich ścieżek — pokazuje rozrzut wyników przy tej samej strategii.
- **`plot_final_capital_distribution`** — histogram kapitału końcowego;
  różnica między średnią a medianą tutaj dobrze pokazuje asymetrię
  rozkładu (grube prawe ogony od rzadkich bardzo dobrych serii).
- **`plot_strategy_comparison`** — nakłada mediany ścieżek dla różnych
  strategii na wspólnym wykresie logarytmicznym — to najbardziej
  bezpośrednia wizualna demonstracja tego, że przy tym samym edge'u
  *position sizing* decyduje o wyniku bardziej niż sam fakt posiadania
  przewagi.

---

## 9. Kluczowe wnioski (podsumowanie)

1. Dodatni edge (p·b > q) jest warunkiem koniecznym, ale nie wystarczającym
   do zarabiania w długim terminie — sposób stawiania (position sizing) jest
   równie krytyczny.
2. Kelly maksymalizuje oczekiwane tempo wzrostu logarytmu kapitału, co w
   praktyce odpowiada maksymalizacji mediany (typowego wyniku), a nie
   wartości oczekiwanej kapitału.
3. Pełny Kelly ma wysoką wariancję — half-Kelly to standardowy kompromis
   między tempem wzrostu a głębokością obsunięć.
4. Strategie typu martingale nie zmieniają fundamentalnej matematyki
   pojedynczego zakładu — tylko przesuwają ryzyko w ogon rozkładu
   (rzadkie, ale katastrofalne straty).
5. Przy stawianiu ułamkowym klasyczne wzory na "ruin probability" nie mają
   zastosowania — trzeba estymować je przez Monte Carlo.
