# Faza 2 — FULL elektrotermalni DC motor

## Cilj

Faza 2 uvodi referentno postrojenje na kome će se kasnije obavljati završni
zero-shot test svih regulatora. Simulator ima jednu skrivenu dinamičku
veličinu — temperaturu namotaja — dok su struja i ugaona brzina merljive.

## Model

Stanje je:

\[
x =
\begin{bmatrix}
i & \omega & T
\end{bmatrix}^{\mathsf T}.
\]

Jednačine su:

\[
\frac{\mathrm di}{\mathrm dt}
=
\frac{u-R(T)i-k_e\omega}{L},
\]

\[
\frac{\mathrm d\omega}{\mathrm dt}
=
\frac{k_ti-b\omega-T_L}{J},
\]

\[
\frac{\mathrm dT}{\mathrm dt}
=
\frac{R(T)i^2-(T-T_{\mathrm{amb}})/R_{\mathrm{th}}}{C_{\mathrm{th}}},
\]

\[
R(T)=R_0[1+\alpha(T-T_0)].
\]

Temperatura ne utiče na \(k_t\), \(k_e\), \(L\), \(J\) ili \(b\). Time je
termički efekat izolovan i može kasnije da se ukloni postavljanjem
\(\alpha=0\).

## Nominalni parametri

| Parametar | Vrednost |
|---|---:|
| \(L\) | 0,003 H |
| \(R_0\) na \(T_0=20^\circ\mathrm C\) | 1,2 Ω |
| \(\alpha\) | 0,00393 \(1/^\circ\mathrm C\) |
| \(k_e\) | 0,08 V·s/rad |
| \(k_t\) | 0,08 N·m/A |
| \(J\) | 0,002 kg·m² |
| \(b\) | 0,002 N·m·s/rad |
| nominalni \(T_L\) | 0,15 N·m |
| \(C_{\mathrm{th}}\) | 20 J/°C |
| \(R_{\mathrm{th}}\) | 1,2 °C/W |
| \(T_{\mathrm{amb}}\) | 25 °C |

Ovo je kontrolisan benchmark parametarski usklađen sa zaključanim domenom od
±48 V, ±25 A, ±500 rad/s i 20–120 °C. Parametri ne predstavljaju identifikovan
komercijalni motor.

## Vremenske konstante i diskretizacija

Nominalne procene su:

\[
\tau_e=\frac{L}{R_0}=2,5\ \mathrm{ms},
\]

\[
\tau_m=
\frac{J}{b+k_tk_e/R_0}
=0,273\ \mathrm{s},
\]

\[
\tau_{\mathrm{th}}=C_{\mathrm{th}}R_{\mathrm{th}}=24\ \mathrm{s}.
\]

Odnos najsporije i najbrže konstante je približno 9600. Kontrolni period je
1 ms, a RK4 korak 0,1 ms, pa električna konstanta sadrži 25 integratorskih
koraka. Početni burn-in je 2 s, što pokriva više od sedam mehaničkih
konstanti, dok osnovna epizoda traje 120 s odnosno pet termičkih konstanti.

## Numerički i bezbednosni interfejs

`ElectrothermalDCMotor.step(u)`:

1. saturiše komandovani napon na \([-48,48]\) V;
2. drži napon i moment opterećenja konstantnim tokom jednog kontrolnog perioda;
3. izvršava deset RK4 podkoraka;
4. prekida simulaciju ako stanje postane nefinite ili napusti dozvoljeni domen;
5. vraća komandovani i primenjeni napon, stanje, saturaciju i razlog prekida.

Fizička stanja se namerno ne odsecaju na granici, jer bi to prikrilo
eksploataciju modela i kršenje ograničenja.

`rollout()` vraća:

- vreme i FULL stanja oblika `(T + 1, 3)`;
- komandovane i primenjene napone;
- korišćene momente opterećenja;
- indikator i razlog ranog prekida.

Randomizovani reset koristi lokalni NumPy generator i za isti seed vraća isto
početno stanje.

## Gate 0

Automatski se proverava:

1. povratak u mirovanje za \(u=0\) i \(T_L=0\);
2. zagrevanje pri protoku struje ka stabilnoj ravnoteži;
3. eksponencijalno hlađenje bez struje;
4. monotoni rast \(R(T)\);
5. manja struja zagrejanog motora pri istom naponu;
6. poklapanje RK4 rešenja za \(h\) i \(h/2\);
7. identični \([i,\omega]\) odzivi za različite temperature kada je
   \(\alpha=0\);
8. jednakost
   \(C_{\mathrm{th}}\dot T=P_{\mathrm{Cu}}-P_{\mathrm{cool}}\).

Pokretanje:

```bash
python -m r2dn_dc_motor.validate_phase2
```

Za JSON izveštaj i PNG sa osnovnim odzivima:

```bash
python -m pip install -e ".[phase2]"
python -m r2dn_dc_motor.validate_phase2 --output-dir results/phase2
```

Uspešan izlaz sadrži `PHASE 2 GATE 0: PASS` i osam pojedinačnih `PASS`
rezultata.
