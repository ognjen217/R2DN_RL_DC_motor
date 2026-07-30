# ADR 0004 — Fizikom informisan probe samo za Gate 1

## Odluka

Vidljivost skrivene temperature proverava se malim ridge regressorom nad
obeležjima izvedenim iz istorije \((i,\omega,u)\) i električkog dinamičkog
reziduala.

## Razlog

Linearan model nad sirovim trenutnim uzorkom ne može pouzdano razdvojiti
termalna stanja. Dinamički rezidual koristi promenu struje kroz vreme i zato
direktno proverava da li istorija sadrži informaciju o promenljivom otporu.

Probe koristi temperature samo kao cilj na trening trajektorijama i za
evaluaciju na odvojenim test trajektorijama. Temperatura nikada nije ulazno
obeležje.

## Ograničenje tvrdnje

Uspeh ovog probe-a dokazuje praktičnu identifikabilnost termičkog režima iz
istorije signala. Ne dokazuje da će R2DN automatski naučiti istu reprezentaciju
i ne predstavlja alternativni konačni world model. Ta tvrdnja se proverava tek
treningom i rollout evaluacijom R2DN-a.
