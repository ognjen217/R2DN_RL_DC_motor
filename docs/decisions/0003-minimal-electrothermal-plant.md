# ADR 0003 — Minimalni elektrotermalni FULL plant

## Status

Prihvaćeno u Fazi 2.

## Kontekst

Glavni eksperiment zahteva sistem sa jednom skrivenom sporom dinamikom koju
R2DN može da zaključi iz istorije struje, brzine i napona. Ako FULL simulator
istovremeno uvede temperaturno zavisan moment, zasićenje, nelinearno trenje i
više termičkih čvorova, poreklo eventualne prednosti rekurentnog modela ne bi
bilo jednoznačno.

## Odluka

FULL plant ima stanje:

\[
x=[i,\omega,T]^{\mathsf T}.
\]

Temperatura utiče samo na otpor armature:

\[
R(T)=R_0[1+\alpha(T-T_0)].
\]

Termički model ima jedan čvor, Joule-ovo zagrevanje i linearno hlađenje prema
konstantnoj temperaturi okoline. Konstante momenta i kontraelektromotorne sile,
trenje i induktivnost ne zavise od temperature.

Napon se saturiše. Struja, brzina i temperatura se ne saturišu; izlazak iz
domena prekida rollout i čuva prvo neispravno stanje i razlog prekida.

Fizički modeli koriste klasični RK4 sa kontrolnim periodom od 1 ms i deset
unutrašnjih koraka od 0,1 ms.

## Posledice

- Skrivena temperatura je jedini izvor memorije koji nedostaje world modelima.
- Ablacija \(\alpha=0\) tačno uklanja uticaj temperature na \([i,\omega]\).
- Simulator nije detaljan model konkretnog komercijalnog motora; on je
  kontrolisan naučni benchmark sa fizički smislenim jedinicama i bilansima.
- Dodatne nelinearnosti pripadaju tek kasnijim stress-testovima.
