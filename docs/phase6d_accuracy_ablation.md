# Phase 6D — eksperimenti za povećanje tačnosti R2DN modela

## Cilj

Faza 6D ispituje da li se dugoročna tačnost može poboljšati bez promene
dozvoljenog interfejsa modela i bez popuštanja kontraktivne parametrizacije.
Polazni Phase-6C rezultat je stabilan tokom 1000 s i 12,92 puta brži od
postojeće FULL/RK4 implementacije, ali ima kombinovani NRMSE 0,3266. Primarni
cilj Faze 6D je kombinovani validation NRMSE manji ili jednak 0,20.

Temperatura namotaja, moment opterećenja, referentna brzina i komandovani napon
i dalje su zabranjeni u treningu. Fit koristi samo train split, izbor koristi
samo validation split, a ID/OOD i stress testovi ostaju isključivo
postselekciona provera.

## Eksperiment 1 — screening postojećih checkpoint-a

Pre novog treninga porede se:

- Phase-6B latent-4/seed-17 checkpoint;
- Phase-6 latent-8/seed-29 checkpoint;
- burn-in dužine 250, 500 i 1000 koraka.

Sve kombinacije predviđaju istih osam validation segmenata u trajanju od 10 s.
Kod dužeg burn-in-a dodaje se starija istorija, dok početak predviđanja i
ciljna trajektorija ostaju nepromenjeni. Zato rezultat razdvaja uticaj
arhitekture od uticaja inicijalizacije latentnog stanja.

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m r2dn_dc_motor.validate_phase6d \
  --screen \
  --profile final \
  --require-cuda \
  --dataset data/phase4-full-v1 \
  --phase6b-checkpoint checkpoints/phase6b/r2dn-v2 \
  --phase6-checkpoint checkpoints/phase6/r2dn-v1 \
  --output-dir results/phase6d
```

Ovaj korak ne menja nijedan checkpoint i ne trenira model. Ako postojeći
latent-8 već dostigne potrebnu tačnost, skupa ablacijska studija može da se
preskoči i model odmah ide na završni 1000 s benchmark. Ako su burn-in rezultati
praktično jednaki, koristi se kraći burn-in radi jednostavnije inicijalizacije.

## Eksperiment 2 — kontrolisana A–D ablacijska studija

| Varijanta | Latent | Burn-in | Curriculum | Izolovani faktor |
|---|---:|---:|---|---|
| A_control | 4 | 250 | do 1000 koraka | kontrolna postavka |
| B_latent8 | 8 | 250 | do 1000 koraka | veći latent |
| C_burnin1000 | 8 | 1000 | do 1000 koraka | duža istorija |
| D_rollout5000 | 8 | 1000 | dodatnih 300 update-a na 5000 koraka | duži rollout loss |

Svaka varijanta se trenira sa seed-ovima 17, 29 i 43. Rangiranje se vrši po
medijani 10.000-koračnog validation NRMSE. Ako je jednostavnija, ranije
deklarisana varijanta unutar 3% od najbolje medijane, bira se jednostavnija
varijanta. Unutar pobedničke varijante čuva se seed sa najmanjim NRMSE.

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m r2dn_dc_motor.validate_phase6d \
  --train \
  --profile final \
  --require-cuda \
  --dataset data/phase4-full-v1 \
  --cache-dir checkpoints/phase6d/run-cache-v1 \
  --checkpoint-dir checkpoints/phase6d/r2dn-v3 \
  --output-dir results/phase6d
```

Cache se ne briše pri prekidu. Ponovljena ista komanda učitava završene
variant/seed run-ove i nastavlja samo nedostajuće. `--overwrite-cache` ne treba
koristiti osim ako se run namerno ponavlja od nule.

## Kriterijumi ispravnosti i uspeha

Protokol je ispravno izvršen ako su završene sve varijante i seed-ovi, svi
rollout-i su konačni, svaki model ima pozitivan contractivity margin i
checkpoint tačno odgovara zaključanom median/tie pravilu. To je odvojeno od
cilja tačnosti: polje `target_met` pokazuje da li je NRMSE pobedničkog seed-a
manji ili jednak 0,20.

Faza generiše:

```text
results/phase6d/phase6d_existing_checkpoint_screen.json
results/phase6d/phase6d_existing_checkpoint_screen.png
results/phase6d/phase6d_accuracy_ablation.json
results/phase6d/phase6d_accuracy_ablation.png
checkpoints/phase6d/r2dn-v3/
```

## Eksperiment 3 — završno poređenje sa RK4

Izabrani checkpoint ponovo prolazi 1000 s Phase-6C benchmark. Benchmark podržava
postojeće Phase-6 i Phase-6B checkpoint-e, kao i novi Phase-6D checkpoint.
Koristi se drugi output direktorijum da se ne prepiše početni latent-4 rezultat.

Ako screening izabere postojeći Phase-6 latent-8, koristi se:

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m r2dn_dc_motor.compare_r2dn_rk4 \
  --require-cuda \
  --dataset data/phase4-full-v1 \
  --checkpoint-dir checkpoints/phase6/r2dn-v1 \
  --phase6b-report results/phase6b/phase6b_latent_and_stability.json \
  --scenario multisine \
  --duration-s 1000 \
  --split validation \
  --anchor-index 0 \
  --chunk-steps 10000 \
  --output-dir results/phase6c_latent8
```

Ako je ipak obavljena puna Phase-6D ablacijska studija, koristi se njen
checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m r2dn_dc_motor.compare_r2dn_rk4 \
  --require-cuda \
  --dataset data/phase4-full-v1 \
  --checkpoint-dir checkpoints/phase6d/r2dn-v3 \
  --phase6b-report results/phase6b/phase6b_latent_and_stability.json \
  --scenario multisine \
  --duration-s 1000 \
  --split validation \
  --anchor-index 0 \
  --chunk-steps 10000 \
  --output-dir results/phase6c_phase6d
```

Ako pobednik koristi burn-in 1000, skripta uzima 750 dodatnih starijih
uzoraka, ali zadržava isto početno fizičko stanje, budući napon i FULL/RK4
referencu. Tako se novi NRMSE i vreme mogu direktno uporediti sa početnim
Phase-6C rezultatom 0,3266 i 18,56 s.
