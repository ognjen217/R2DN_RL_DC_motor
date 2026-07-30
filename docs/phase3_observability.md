# Faza 3 — vidljivost skrivene temperature

## Cilj

Faza 3 proverava da li je skrivena temperatura namotaja istovremeno:

1. dovoljno uticajna da menja merljive trajektorije i kvalitet upravljanja;
2. mnogo veća od numeričke greške RK4 solvera;
3. inferabilna iz istorije dozvoljenih signala \((i,\omega,u)\);
4. netrivijalna za procenu iz samo jednog trenutnog uzorka.

Ovo je go/no-go korak pre generisanja konačnog skupa podataka. U ovoj fazi se
ne trenira R2DN i ne čuva se dataset Faze 4.

## Upareni hladan/topao eksperiment

Dve FULL simulacije počinju sa istim:

\[
i(0)=0,\qquad \omega(0)=0,
\]

i dobijaju potpuno isti piecewise-constant naponski signal i moment
opterećenja. Razlikuju se samo po:

\[
T_\mathrm{cold}(0)=25^\circ\mathrm{C},\qquad
T_\mathrm{hot}(0)=90^\circ\mathrm{C}.
\]

Mere se maksimalne razlike struje, brzine, otpora i ubrzanja. Isti eksperiment
se ponavlja sa dvostruko manjim RK4 korakom. Termički signal se normira
zaključanim granicama struje i brzine i poredi sa razlikom osnovnog i finog
solvera.

## Dijagnostički PI eksperiment

Hladan i topao motor prate istu step referencu brzine pomoću istog PI
regulatora. Ovaj regulator nije konačni PI baseline projekta i ne koristi se za
RL. Služi samo da kvantifikuje:

- IAE brzine;
- kumulativni upravljački napor \(\int |u(t)|\,\mathrm dt\);
- maksimalni potreban napon;
- bezbednost obe putanje.

## Pilot za dijagnostičku identifikabilnost

Pilot sadrži 16 determinističkih trajektorija od po 3 s. Početne temperature
ravnomerno pokrivaju interval od 25 do 90 °C. Napon uzima vrednosti iz skupa
\(\{-18,-9,0,9,18\}\) V i menja se na 100 ms. Podela se vrši po celim
trajektorijama: 12 trening i 4 test putanje. Test temperature pokrivaju ceo
opseg.

Temperatura se koristi isključivo kao trening cilj i za evaluaciju. Nijedna
temperaturna vrednost nije deo obeležja.

## History probe

Pomoćni probe koristi istoriju dozvoljenih signala i električnu jednačinu da
formira dinamički rezidual:

\[
q_k =
u_k-L\frac{i_{k+1}-i_k}{T_s}
-k_e\frac{\omega_{k+1}+\omega_k}{2}.
\]

U kratkom prozoru spora promena temperature dozvoljava procenu efektivnog
otpora iz odnosa \(q\approx R(T)i\). Iz istorije se računaju:

- least-squares i medijanska procena otpora;
- RMS i srednja apsolutna struja;
- srednja brzina;
- srednji napon.

Nad ovih šest obeležja trenira se mali ridge regressor za dijagnostičku
procenu temperature. Ovo je namerno fizikom informisan dokaz da istorija
sadrži termičku informaciju; nije deo R2DN-a niti kandidat za konačni world
model.

Za poređenje se trenira isti tip ridge regressora samo nad trenutnim
\((i_k,\omega_k,u_{k-1})\). Gate zahteva da history probe bude precizan na
potpuno novim trajektorijama, ali i da trenutni probe ostane nedovoljno
precizan. Time se potvrđuje potreba za memorijom.

## Gate 1

Gate prolazi samo ako su ispunjeni svi uslovi iz `configs/phase3.toml`:

- odnos termičkog signala i RK4 greške je najmanje \(10^4\);
- maksimalna razlika struje je najmanje 0,5 A;
- maksimalna razlika brzine je najmanje 5 rad/s;
- temperatura menja IAE za najmanje 5%;
- temperatura menja upravljački napor za najmanje 3%;
- trenutni probe ima MAE od najmanje 5 °C;
- history probe sa 250 ms istorije ima MAE najviše 2 °C;
- history probe poboljšava MAE za najmanje 50%;
- test sadrži najmanje 100 uzoraka;
- nema curenja temperature, preklapanja trajektorija ni prekida simulatora.

Pragovi su konzervativni u odnosu na očekivani fizički signal i služe da
spreče nastavak projekta ako termički efekat nije praktično relevantan.
