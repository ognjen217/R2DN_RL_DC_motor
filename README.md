# R2DN world model za upravljanje DC motorom

Eksperimentalni kod za ispitivanje da li kontraktivni rekurentni model R2DN
omogućava bolji transfer RL regulatora od nepotpunog izotermskog fizičkog
modela kada DC motor ima skrivenu elektrotermalnu dinamiku.

## Trenutni status

Implementirane su:

- **Faza 0** — zaključana eksperimentalna postavka, signali, modeli i metrike;
- **Faza 1** — projektovan i izvršno validiran R2DN interfejs;
- **Faza 2** — FULL elektrotermalni DC motor, RK4 i Gate 0 validacija;
- **Faza 3** — Gate 1 vidljivosti skrivene temperature i dijagnostički history probe;
- **Faza 4** — verzionisan FULL dataset, disjunktne whole-trajectory podele,
  train-only normalizacija i provere integriteta;
- **Faza 5** — `ISO-NOM` i globalno kalibrisani `ISO-CAL`, checkpoint vezan
  za dataset i one-step/kratka/srednja/duga ID/OOD evaluacija;
- **Faza 6** — curriculum trening zvaničnog R2DN-a, pilot latentnih dimenzija,
  ponavljanje za više seed-ova i verzionisan checkpoint vezan za dataset;
- time-major format sekvenci i temperature-free model view;
- observation burn-in i autoregresivni free-rollout adapter;
- verzionisan format checkpoint manifesta;
- integracioni test sa zvaničnim `ContractingR2DN` kodom;
- provera pozitivnosti reziduala uslova (20) iz R2DN rada;
- deterministički reset, saturacija napona i dijagnostika prekida plant rollout-a;
- analiza električne, mehaničke i termičke vremenske konstante;
- osam fizičkih i numeričkih testova FULL simulatora;
- upareni hladan/topao eksperiment sa identičnim \((i_0,\omega_0,u)\);
- poređenje termičkog signala sa greškom RK4 usitnjavanja;
- dijagnostički PI test uticaja temperature na praćenje i upravljački napor;
- temperature-free pomoćni regressor sa podelom po celim trajektorijama.
- osam porodica pobude za identifikaciju i buduće upravljanje;
- `ci` profil sa 32 trajektorije i zaključani `final` profil sa 320
  trajektorija i 4,8 miliona prelaza;
- odvojeni ID i OOD test skupovi sa višom temperaturom, jačim opterećenjem,
  novim profilima i promenama fizičkih parametara;
- SHA-256 identitet svake trajektorije i fingerprint kompletnog dataseta;
- sirovi/evaluacioni pogled \((i,\omega,T,u,T_L,r)\) i strogo odvojeni
  R2DN pogled \((i,\omega,u)\).

RL još nije implementiran. Finalni `ISO-CAL` i R2DN checkpoint-i namerno se
generišu lokalno iz korisnikovog Phase-4 dataseta.

## Brza provera

Potreban je Python 3.11 ili noviji.

```bash
python -m pip install -e ".[dev]"
python -m r2dn_dc_motor.validate_phase0
python -m r2dn_dc_motor.validate_phase1
python -m r2dn_dc_motor.validate_phase2
python -m r2dn_dc_motor.validate_phase3
python -m r2dn_dc_motor.validate_phase4 --spec-only
python -m r2dn_dc_motor.validate_phase5 --spec-only
python -m r2dn_dc_motor.validate_phase6 --spec-only
pytest -v
ruff check .
```

Za proveru stvarnog pinned upstream backend-a:

```bash
python -m pip install -e ".[dev,r2dn]"
pytest -v -m r2dn_integration
```

Očekivani rezultat validatora je:

```text
PHASE 0: PASS
PHASE 1: PASS
PHASE 2 SPEC: PASS
PHASE 2 GATE 0: PASS
PHASE 3 SPEC: PASS
PHASE 3 GATE 1: PASS
PHASE 4 SPEC: PASS
PHASE 5 SPEC: PASS
PHASE 6 SPEC: PASS
```

Za generisanje Phase-2 JSON izveštaja i PNG grafikona:

```bash
python -m pip install -e ".[dev,phase2]"
python -m r2dn_dc_motor.validate_phase2 --output-dir results/phase2
```

Za generisanje Phase-3 JSON izveštaja i PNG dijagnostike:

```bash
python -m pip install -e ".[dev,phase3]"
python -m r2dn_dc_motor.validate_phase3 --output-dir results/phase3
```

Za brzu izgradnju i potpunu proveru Phase-4 `ci` dataseta:

```bash
python -m pip install -e ".[dev,phase4]"
python -m r2dn_dc_motor.validate_phase4 \
  --generate \
  --profile ci \
  --output-dir data/phase4-ci \
  --artifacts-dir results/phase4-ci
```

Za zaključani dataset od 320 trajektorija i 4,8 miliona prelaza:

```bash
python -m r2dn_dc_motor.validate_phase4 \
  --generate \
  --profile final \
  --output-dir data/phase4-full-v1 \
  --artifacts-dir results/phase4-final
```

Za jedan globalni `ISO-CAL` fit i kompletnu Phase-5 evaluaciju:

```bash
python -m pip install -e ".[dev,phase5]"
python -m r2dn_dc_motor.validate_phase5 \
  --fit \
  --dataset data/phase4-full-v1 \
  --checkpoint checkpoints/phase5/iso_cal.json \
  --output-dir results/phase5
```

Komanda koristi sve cele train trajektorije, ali ne čita temperaturu ni
stvarno opterećenje tokom kalibracije. Bez `--fit` koristi već postojeći
checkpoint i samo ponavlja evaluaciju.

Za finalni R2DN pilot, curriculum trening sa tri seed-a i checkpoint:

```bash
python -m pip install -e ".[dev,phase6,phase6-cuda12]"
python -m r2dn_dc_motor.validate_phase6 --runtime-only --require-cuda
python -m r2dn_dc_motor.validate_phase6 \
  --train \
  --profile final \
  --require-cuda \
  --dataset data/phase4-full-v1 \
  --checkpoint-dir checkpoints/phase6/r2dn-v1 \
  --output-dir results/phase6
```

Komanda poredi latentne dimenzije 4/6/8 u pilotu, izabranu arhitekturu trenira
sa seed-ovima 17/29/43 i bira checkpoint sa najmanjim validation free-rollout
NRMSE. CUDA preflight mora da prikaže `backend: gpu` i `CUDA active: yes`;
`--require-cuda` sprečava tihi CPU fallback. Temperatura, opterećenje, referenca
i komandovani napon nikad ne ulaze u trening. Ovaj validator potvrđuje trening
i integritet checkpoint-a; čisto poređenje sa ISO-CAL i uslov za prelazak na RL
pripadaju Fazi 7.

## Struktura

```text
.
├── configs/
│   ├── phase0.toml
│   ├── phase1.toml
│   ├── phase2.toml
│   ├── phase3.toml
│   ├── phase4.toml
│   ├── phase5.toml
│   └── phase6.toml
├── docs/
│   ├── phase0_specification.md
│   ├── phase1_design.md
│   ├── phase2_simulator.md
│   ├── phase3_observability.md
│   ├── phase4_dataset.md
│   ├── phase5_baselines.md
│   ├── phase6_r2dn_training.md
│   ├── references.md
│   └── decisions/
├── src/r2dn_dc_motor/data/
│   ├── phase4_dataset.py
│   ├── phase4_generation.py
│   ├── r2dn_windows.py
│   └── sequences.py
├── src/r2dn_dc_motor/numerics/
│   └── rk4.py
├── src/r2dn_dc_motor/plants/
│   ├── electrothermal.py
│   └── isothermal.py
├── src/r2dn_dc_motor/validation/
│   ├── phase2.py
│   ├── phase3.py
│   ├── phase4.py
│   ├── phase5.py
│   └── phase6.py
├── src/r2dn_dc_motor/models/
│   ├── checkpoint.py
│   ├── isothermal_calibration.py
│   ├── r2dn_adapter.py
│   ├── r2dn_training.py
│   └── temperature_probe.py
├── src/r2dn_dc_motor/
│   ├── phase1_spec.py
│   ├── phase2_spec.py
│   ├── phase3_spec.py
│   ├── phase4_spec.py
│   ├── phase5_spec.py
│   ├── phase6_spec.py
│   ├── spec.py
│   ├── validate_phase0.py
│   ├── validate_phase1.py
│   ├── validate_phase2.py
│   ├── validate_phase3.py
│   ├── validate_phase4.py
│   ├── validate_phase5.py
│   └── validate_phase6.py
└── tests/
```

Konfiguracije u `configs/` su mašinski čitljivi izvori istine. Testovi odbijaju
curenje temperature, drift interfejsa i fizičkog domena, teacher forcing u
slobodnom rollout-u, nedovoljno fin RK4 korak i checkpoint napravljen drugim
upstream commitom.

Gate 1 dodatno odbija sample-level podelu pilot podataka, temperaturu u
ulaznim obeležjima, termički signal uporediv sa numeričkom greškom i
identifikabilnost koja postoji samo u trenutnom uzorku bez istorije.

Faza 4 odbija bilo koji izvor osim FULL simulatora, preklapanje trajektorija
između podela, nepotpune ili promenjene fajlove, normalizaciju koja nije
izračunata isključivo nad `train`, curenje temperature/opterećenja u R2DN
pogled i OOD skup koji nije stvarno izvan trening domena.

Faza 5 odbija per-episode kalibraciju, korišćenje temperature ili stvarnog
opterećenja u fitu, menjanje parametara na validation/ID/OOD podelama,
nepotpun train skup i checkpoint sa fingerprintom drugog dataseta.

Faza 6 odbija curenje evaluation-only feature-a, fit van train podele, izbor
checkpoint-a na ID/OOD skupu, teacher forcing u rollout loss-u, promenjenu
train-only normalizaciju, nekompletan curriculum, nekonačne loss/gradijent
vrednosti i checkpoint sa promenjenim parametrima, istorijom ili upstream
commitom.

## R2DN referenca

Koristi se zvanična JAX implementacija iz
[`nic-barbara/R2DN`](https://github.com/nic-barbara/R2DN), vezana za konkretan
commit u konfiguraciji. Kod se instalira kao pinned opciona zavisnost i nije
kopiran u ovaj repozitorijum.

R2DN je u referentnom radu diskretni rekurentni state-space model. Zbog toga će
R2DN backend koristiti diskretne korake, dok će fizički ODE modeli koristiti
RK4 između kontrolnih trenutaka.

Važno: upstream garancija kontraktivnosti važi za latentne putanje sa istom
spoljašnjom ulaznom sekvencom. Pošto naš free rollout vraća predikovani izlaz
na ulaz sledećeg koraka, stabilnost cele autoregresivne petlje proverava se
posebno numeričkim rollout testovima i ne predstavlja se kao formalna posledica
same upstream garancije.
