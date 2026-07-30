# ADR-0002: Autoregresivni R2DN sa observation burn-in intervalom

- Status: prihvaćeno
- Datum: 2026-07-30

## Kontekst

Zvanični `ContractingR2DN` ima latentno stanje, eksplicitni ulaz i sekvencijalni
rollout, ali nema poseban encoder istorije niti gotov autoregresivni world-model
interfejs. Skrivena temperatura se ne može pouzdano odrediti iz jednog para
\((i,\omega)\).

## Odluka

- R2DN regressor je `[i_k, omega_k, u_k]`.
- Target je `[i_{k+1}, omega_{k+1}]`.
- Na početku epizode nulto latentno stanje prolazi kroz observation burn-in.
- Posle burn-in intervala predikovani izlaz se vraća kao opservacioni deo
  sledećeg regressora.
- Buduće stvarne opservacije se ne koriste u slobodnom rollout-u.
- Pilot latentna dimenzija je 4, dok će konačna vrednost biti hiperparametar.

## Posledice

Model može iz merene istorije formirati latentni termički režim bez direktnog
pristupa temperaturi. Isti adapter podržava one-step trening, burn-in i slobodnu
višekoračnu evaluaciju.

Kontraktivnost zvaničnog R2DN-a važi za identične spoljašnje ulazne sekvence.
Autoregresivna povratna sprega zato zahteva zasebne numeričke testove stabilnosti
i neće se predstavljati kao formalno garantovana samim R2DN uslovom.
