# Module 2: Random Walk → Brownian Motion → GBM — Theory

Ten dokument wyjaśnia matematykę modułu 2 i mapuje ją na kod w
`src/quant_sims/stochastic_processes/`. Kontynuacja `theory.md` (moduł 1).

---

## 1. Simple Random Walk

Suma niezależnych kroków ±1 (lub ogólniej ±step_size):

```
S_n = X_1 + X_2 + ... + X_n,   X_i ∈ {-step, +step}
```

Dla wersji nieobciążonej (p=0.5): `E[S_n] = 0`, `Var(S_n) = n·step²`.
Dla wersji obciążonej (p ≠ 0.5): `E[X_i] = step·(2p−1)`, więc
`E[S_n] = n·step·(2p−1)`.

To bezpośrednio ten sam mechanizm co rzut monetą z modułu 1 — różnica jest
w tym, *co* sumujemy: tam kapitał rósł **multiplikatywnie** (iloczyn
czynników `1±f·b`), tutaj `S_n` rośnie **addytywnie** (suma kroków). Stąd w
module 1 potrzebny był logarytm żeby zamienić iloczyn na sumę (Kelly); tutaj
suma jest już naturalną strukturą procesu.

**Kod:** `RandomWalk.simulate()` / `simulate_paths()` w `random_walk.py`
generują `moves` jako wektor ±step_size (przez `np.where` na losowych
liczbach porównanych z `p_up`), potem `np.cumsum` daje ścieżkę `S_n`.

Testy (`test_random_walk.py`) sprawdzają te wzory statystycznie: generujemy
5000 niezależnych ścieżek i porównujemy empiryczną średnią/wariancję
kapitału końcowego z wartościami teoretycznymi (z tolerancją, bo to
estymacja Monte Carlo, nie dokładna liczba).

---

## 2. Twierdzenie Donskera — most do Brownian Motion

### 2.1 Dlaczego random walk "staje się" ciągły

Random walk to proces **dyskretny** — skacze co krok. Ale gdy przeskalujemy
go odpowiednio i puścimy liczbę kroków do nieskończoności, granica jest
procesem **ciągłym**. To jest właśnie **twierdzenie Donskera** (funkcjonalna
wersja centralnego twierdzenia granicznego — CLT, ale dla całych *trajektorii*,
nie tylko pojedynczej wartości końcowej).

Formalnie: zdefiniujmy przeskalowany proces na siatce czasu `t ∈ [0, T]`:

```
W_n(t) = S_⌊n·t⌋ / √n
```

Gdy `n → ∞`, `W_n(t)` zbiega w rozkładzie (na przestrzeni funkcji ciągłych)
do **standardowego procesu Wienera** `W(t)`.

### 2.2 Skąd bierze się dokładnie `√n`

To nie przypadek — to konsekwencja CLT. Wariancja pojedynczego kroku wynosi
`step²` (stała), więc `Var(S_n) = n · step²` rośnie liniowo z `n`. Żeby
przeskalowana zmienna miała **skończoną, ustaloną wariancję** w granicy
(a nie rozbiegała się do nieskończoności albo kolapsowała do zera),
musimy podzielić przez coś, co rośnie jak `√n` — bo wtedy:

```
Var(S_n / √n) = Var(S_n) / n = step² · n / n = step²  (stała, niezależna od n)
```

To dokładnie ten sam mechanizm, co standaryzacja w zwykłym CLT
(`(X̄ − μ)/(σ/√n)`), tylko zastosowany do całej trajektorii naraz, a nie
jednej liczby.

**Kod:** `RandomWalk.scaling_limit_paths()` w `random_walk.py` implementuje
dokładnie `W_n(t) = S_⌊nt⌋/√n` (metoda celowo wymusza p_up=0.5 i step=1,
bo standardowy proces Wienera wymaga symetrycznych przyrostów o wariancji 1
— patrz docstring w kodzie).

Test `test_scaling_limit_paths_shape_and_variance` sprawdza, że przy
`n_steps=1000` rozkład wartości końcowej `W_n(T)` ma empiryczną wariancję
bliską `T` (zgodnie z tym, że `W(T) ~ N(0, T)` dla standardowego procesu
Wienera) — a notebook dodatkowo *wizualnie* pokazuje tę zbieżność, rysując
przeskalowane ścieżki dla n = 10, 100, 1000, 10000 obok siebie: im większe
n, tym "gładsza" i bardziej ciągła wygląda trajektoria.

---

## 3. Standardowy proces Wienera — definicja formalna

`W(t)` jest standardowym procesem Wienera, jeśli spełnia cztery warunki:

1. `W(0) = 0`
2. Przyrosty niezależne: `W(t) − W(s)` niezależne od historii przed `s`,
   dla `t > s`
3. Przyrosty gaussowskie: `W(t) − W(s) ~ N(0, t − s)`
4. Ścieżki ciągłe (prawie na pewno), ale **nigdzie nieróżniczkowalne** —
   to matematyczna konsekwencja tego, że przyrosty skalują się jak `√dt`,
   a nie `dt` (stąd "postrzępiony", fraktalny wygląd trajektorii nawet w
   powiększeniu)

### 3.1 Symulacja jest dokładna, nie przybliżona

Dzięki własności (3) możemy symulować `W(t)` na dowolnej siatce czasowej
**bez błędu numerycznego** — wystarczy losować niezależne przyrosty
gaussowskie o odpowiedniej wariancji i sumować kumulacyjnie:

```
W(t_{k+1}) = W(t_k) + √dt · Z_k,     Z_k ~ N(0,1) i.i.d.
```

To działa, bo `Var(przyrostu na odcinku dt) = dt` z definicji (3) — nie
musimy niczego przybliżać metodami typu Euler-Maruyama (te są potrzebne
dopiero dla bardziej złożonych SDE, gdzie nie ma rozwiązania w postaci
zamkniętej — patrz przyszłe moduły: Vasicek, Heston).

### 3.2 Dryft i zmienność (arithmetic Brownian motion)

Uogólniona wersja z dryftem `μ` i skalą `σ`:

```
X(t) = X(0) + μt + σW(t)
```

ma `E[X(t)] = X(0) + μt` i `Var(X(t)) = σ²t`, czyli `Std(X(t)) = σ√t` —
**skalowanie pierwiastkiem z czasu**, kluczowa i często cytowana własność
procesów Wienera (stąd np. "zmienność roczna = zmienność dzienna × √252").

**Kod:** `BrownianMotion.simulate_path()` / `simulate_paths()` w
`brownian_motion.py` implementują dokładnie ten wzór — `dt = T/n_steps`,
przyrosty `rng.normal(0, √dt)`, kumulacja przez `np.cumsum`. Statyczne
metody `theoretical_mean()` i `theoretical_std()` dają wzory referencyjne
używane w testach i w notebooku do porównania z symulacją.

---

## 4. Geometric Brownian Motion (GBM)

### 4.1 Dlaczego zwykły proces Wienera nie nadaje się do modelowania cen

`X(t) = X(0) + μt + σW(t)` może przyjąć dowolną wartość, łącznie z ujemną.
Cena akcji nie może być ujemna. Rozwiązanie: modelować nie samą cenę, tylko
jej **zmiany relatywne** (stopy zwrotu), przez SDE:

```
dS_t = μ·S_t dt + σ·S_t dW_t
```

Interpretacja: `μ` i `σ` to **relatywna** (procentowa) prędkość dryftu i
zmienności — dokładnie tak, jak myślimy o zwrotach z akcji ("+8% rocznie",
"25% zmienności"), nie o zmianie w jednostkach walutowych.

### 4.2 Lemat Itô — jak dostajemy rozwiązanie w zamkniętej formie

To SDE nie ma trywialnego rozwiązania wprost (bo `dS_t` zależy od samego
`S_t`). Trik: zastosuj **lemat Itô** (odpowiednik reguły łańcuchowej dla
rachunku stochastycznego) do `f(S) = log(S)`:

```
d(log S_t) = (1/S_t)dS_t − (1/2)(1/S_t²)(dS_t)²
```

Podstawiając `dS_t = μS_t dt + σS_t dW_t` i korzystając z reguły Itô
`(dW_t)² = dt` (kluczowa różnica względem zwykłego rachunku różniczkowego!):

```
d(log S_t) = (μ − σ²/2) dt + σ dW_t
```

To już jest zwykłe arithmetic Brownian motion (sekcja 3.2) dla `log(S_t)`,
więc możemy je scałkować wprost:

```
log(S_t) = log(S_0) + (μ − σ²/2)t + σW(t)
S_t = S_0 · exp[(μ − σ²/2)t + σW(t)]
```

To jest **dokładne** rozwiązanie (żadnej dyskretyzacji/przybliżenia) —
możemy je ewaluować w dowolnym punkcie czasu, potrzebujemy tylko wygenerować
`W(t)` (sekcja 3.1).

**Ważne:** człon `−σ²/2` (tzw. "Itô correction" albo "volatility drag") to
nie błąd — to konsekwencja tego, że `E[exp(X)] ≠ exp(E[X])` dla zmiennej
losowej `X` (nierówność Jensena). Gdyby go pominąć, `E[S_t]` wyszłoby
błędne.

### 4.3 Rozkład i momenty

`log(S_t/S_0) ~ N((μ−σ²/2)t, σ²t)`, czyli `S_t` ma **rozkład log-normalny**:

```
E[S_t] = S_0 · exp(μt)
Var[S_t] = S_0² · exp(2μt) · (exp(σ²t) − 1)
```

**Kod:** `GeometricBrownianMotion.simulate_paths()` w `gbm.py` generuje
`W(t)` (kumulacja przyrostów gaussowskich, jak w Brownian motion), potem
podstawia wprost do wzoru `S_t = S_0·exp[(μ−σ²/2)t + σW(t)]` — stąd
komentarz w kodzie, że to "exact closed-form", nie Euler-Maruyama.
`theoretical_mean()` / `theoretical_variance()` implementują wzory powyżej,
używane w testach (`test_mean_matches_theory`) do sprawdzenia zgodności
symulacji z teorią.

### 4.4 Kalibracja do realnych danych (`fit_mu_sigma`)

Mając obserwowany szereg cen, chcemy oszacować `μ` i `σ`, które "najlepiej"
opisują ten szereg jako GBM. Metoda **momentów na log-returns**:

```
r_i = log(S_{i+1}/S_i)  — te przyrosty mają rozkład N((μ−σ²/2)dt, σ²dt)
```

Stąd:
```
σ̂ = std(r) / √dt
μ̂ = mean(r) / dt + σ̂²/2
```

**Kod:** `fit_mu_sigma()` w `gbm.py` implementuje dokładnie te dwa wzory.
Test `test_fit_mu_sigma_recovers_known_parameters` symuluje ścieżkę o
znanych `μ, σ`, potem "zapomina" te parametry i odtwarza je z samej ceny —
sprawdzając, że estymator faktycznie działa. Zauważ w teście dużo większą
tolerancję dla `μ̂` niż `σ̂` — to nieprzypadkowe: `μ̂` ma znacznie wyższą
wariancję jako estymator (potrzeba dekad danych, żeby dobrze oszacować
dryft; zmienność szacuje się dużo szybciej i precyzyjniej) — to znany,
praktyczny problem w finansach ilościowych, nie błąd implementacji.

---

## 5. GBM vs. rzeczywistość — dlaczego to tylko przybliżenie

Model GBM zakłada, że log-returns są **dokładnie normalne**. Realne dane
giełdowe systematycznie łamią to założenie:

- **Fat tails (grube ogony)**: ekstremalne ruchy cen (krachy, gwałtowne
  wzrosty) zdarzają się częściej, niż przewiduje rozkład normalny.
  Kwantyfikuje to **nadmiarowa kurtoza** (`excess kurtosis`) — dla
  dokładnego rozkładu normalnego wynosi 0; dla realnych danych giełdowych
  zwykle wyraźnie dodatnia.
- **Zmienność nie jest stała**: `σ` w realnym rynku zmienia się w czasie
  (tzw. "volatility clustering" — okresy spokoju i okresy turbulencji), a
  GBM zakłada `σ` jako stałą.
- **Skoki**: realne ceny czasem "skaczą" (np. po ogłoszeniach wyników
  finansowych) zamiast poruszać się w sposób ciągły, jak zakłada Brownian
  motion.

**Kod/notebook:** sekcja 5 notebooka `02_random_walk_to_brownian.ipynb`
pobiera realne dane S&P 500 (`yfinance`), liczy log-returns, nakłada
histogram na dopasowany rozkład normalny, i liczy `excess kurtosis` przez
`scipy.stats.kurtosis()` — wartość istotnie dodatnia potwierdza fat tails
wizualnie i liczbowo.

To bezpośrednio motywuje kolejne moduły w roadmapie:
- **Jump diffusion (Merton)** — dodaje skoki do GBM
- **Stochastic volatility (Heston)** — pozwala `σ` zmieniać się losowo
  w czasie zamiast być stałą

---

## 6. Podsumowanie (mapa pojęć)

| Koncept | Wzór | Gdzie w kodzie |
|---|---|---|
| Random walk | `S_n = ΣX_i` | `RandomWalk.simulate_paths` |
| Skalowanie do Browna | `W_n(t) = S_⌊nt⌋/√n` | `RandomWalk.scaling_limit_paths` |
| Standardowy proces Wienera | `W(t)−W(s) ~ N(0,t−s)` | `BrownianMotion.simulate_paths` |
| Dryft + zmienność | `X(t)=X0+μt+σW(t)` | `BrownianMotion` z `mu`, `sigma` |
| GBM (SDE) | `dS=μS dt+σS dW` | — (definiujący model) |
| GBM (rozwiązanie) | `S_t=S_0 exp[(μ−σ²/2)t+σW(t)]` | `GeometricBrownianMotion.simulate_paths` |
| Kalibracja GBM | `σ̂=std(r)/√dt`, `μ̂=mean(r)/dt+σ̂²/2` | `GeometricBrownianMotion.fit_mu_sigma` |
| Test modelu | excess kurtosis realnych log-returns | notebook, sekcja 5 |
