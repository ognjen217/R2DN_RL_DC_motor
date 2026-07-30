# ADR-0001: Različita unutrašnja vremenska reprezentacija uz zajednički interfejs

- Status: prihvaćeno
- Datum: 2026-07-29

## Kontekst

Referentni elektrotermalni motor i izotermski baseline modeli prirodno su
kontinualni ODE sistemi. Zvanični R2DN rad i kod definišu model u diskretnom
vremenu.

## Odluka

- FULL, ISO-NOM i ISO-CAL koriste ODE + RK4.
- R2DN koristi diskretni rekurentni korak.
- Svi modeli napreduju za isti spoljašnji kontrolni period.
- R2DN se neće prepravljati u neuralni ODE u osnovnom eksperimentu.

## Posledice

Poređenje ostaje fer na nivou ulazno-izlaznih sekvenci i policy interfejsa.
Fizički modeli mogu imati više unutrašnjih RK4 podkoraka, dok R2DN obavlja
jedan naučeni diskretni prelaz. Time se zadržava vernost zvaničnoj R2DN
implementaciji i izbegava dodatna, nedokazana arhitektonska izmena.

