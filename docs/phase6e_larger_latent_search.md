# Phase 6E — puni trening većih latentnih dimenzija

## Zašto je potreban novi eksperiment

Phase 6B jeste uključivao latentne dimenzije 10, 12 i 16, ali samo u kratkom
pilot-treningu. Posle pilotskog izbora, puni Phase-6 curriculum sa seed-ovima
17, 29 i 43 izvršen je samo za izabranu latentnu dimenziju. Zato prethodni
rezultati ne odgovaraju na pitanje da li bi latent 12 ili 16, posle punog
treninga, bio precizniji od latent-8 modela.

Phase 6E prvi put trenira latentne dimenzije 8, 12 i 16 pod potpuno istim
uslovima. Ovo je devet punih treninga, po tri za svaku dimenziju. Svaki run
koristi:

- zvanični `ContractingR2DN` sa kontraktivnom polar parametrizacijom;
- samo normalizovanu struju, brzinu i primenjeni napon;
- burn-in od 250 koraka;
- iste hidden slojeve `[32, 32]` i feature širinu 32;
- isti AdamW optimizer i loss;
- isti Phase-6 curriculum sa rollout horizontima 1, 10, 100 i 1000;
- isti train-only normalization i iste seed-ove 17, 29 i 43.

Jedini eksperimentalni faktor je latentna dimenzija.

## Postupak izbora

Fiksni Phase-4 validation prozori i dalje se računaju za svaki run, ali
arhitekturu u Phase 6E biraju tri unapred zaključana `multisine` rollout-a od
po 100 s. Svaki scenario koristi drugi validation anchor i poredi R2DN sa
FULL/RK4 referencom. Te pobude nisu jednake kanonskoj Phase-6C pobudi.

Za svaki run računaju se medijane kombinovanog NRMSE-a, NRMSE-a struje i
NRMSE-a brzine kroz tri scenarija. Za svaki latent zatim se računa medijana
kroz seed-ove 17, 29 i 43. Bira se najmanji latent čija je kombinovana medijana
unutar 3% od najbolje. Struja i brzina se u izveštaju prikazuju odvojeno, iako
je kombinovani NRMSE primarna selekciona metrika.

Kanonski 1000 s Phase-6C `multisine` scenario ne učestvuje u izboru i ostaje
završni test samo jednog pobedničkog checkpoint-a.

## Pokretanje

Posle primene Phase-6E paketa instalirati ažurirani projekat i proveriti
specifikaciju i CUDA runtime:

```bash
source .venv/bin/activate
python -m pip install -e ".[dev,phase6,phase6-cuda12]"

python -m r2dn_dc_motor.validate_phase6e --spec-only --profile final

CUDA_VISIBLE_DEVICES=0 \
python -m r2dn_dc_motor.validate_phase6e \
  --runtime-only \
  --profile final \
  --require-cuda
```

Puni resumable eksperiment pokreće se komandom:

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m r2dn_dc_motor.validate_phase6e \
  --train \
  --profile final \
  --require-cuda \
  --dataset data/phase4-full-v1 \
  --phase6b-report results/phase6b/phase6b_latent_and_stability.json \
  --cache-dir checkpoints/phase6e/run-cache-v1 \
  --checkpoint-dir checkpoints/phase6e/r2dn-v1 \
  --output-dir results/phase6e
```

Svaki završeni run se odmah čuva u cache-u. Ako se proces prekine pre kraja,
ponoviti istu komandu; završeni latent/seed run-ovi neće se trenirati ponovo.
Opcija `--overwrite-cache` koristi se samo kada je namerno potrebno ponoviti
sve treninge od početka.

Rezultati su:

```text
results/phase6e/phase6e_larger_latent_search.json
results/phase6e/phase6e_larger_latent_search.png
checkpoints/phase6e/r2dn-v1/
```

`PASS` znači da su završene sve kombinacije, da su trening i svi selection
rollout-i konačni, da je svaki contractivity margin pozitivan i da je
checkpoint izabran zaključanim median/tie pravilom. Polje `target_met` odvojeno
pokazuje da li je kombinovana troseedna medijana manja ili jednaka 0,20.

## Završni 1000 s test

Samo izabrani Phase-6E checkpoint prolazi kanonski milion-koračni benchmark:

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m r2dn_dc_motor.compare_r2dn_rk4 \
  --require-cuda \
  --dataset data/phase4-full-v1 \
  --checkpoint-dir checkpoints/phase6e/r2dn-v1 \
  --phase6b-report results/phase6b/phase6b_latent_and_stability.json \
  --scenario multisine \
  --duration-s 1000 \
  --split validation \
  --anchor-index 0 \
  --chunk-steps 10000 \
  --output-dir results/phase6c_phase6e
```

Pobednik se smatra praktično boljim od postojećeg latent-8/seed-29 modela samo
ako popravi kombinovani NRMSE bez neprihvatljivog pogoršanja jedne izlazne
veličine i ostane konačan tokom svih milion koraka. Veća latentna dimenzija sama
po sebi nije dokaz boljeg modela.
