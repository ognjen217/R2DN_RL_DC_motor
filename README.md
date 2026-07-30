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

Konačni dataset, trening R2DN-a i RL još nisu implementirani. Pilot iz Faze 3
postoji samo u memoriji tokom validacije i ne predstavlja dataset Faze 4.

## Brza provera

Potreban je Python 3.11 ili noviji.

```bash
python -m pip install -e ".[dev]"
python -m r2dn_dc_motor.validate_phase0
python -m r2dn_dc_motor.validate_phase1
python -m r2dn_dc_motor.validate_phase2
python -m r2dn_dc_motor.validate_phase3
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

## Struktura

```text
.
├── configs/
│   ├── phase0.toml
│   ├── phase1.toml
│   ├── phase2.toml
│   └── phase3.toml
├── docs/
│   ├── phase0_specification.md
│   ├── phase1_design.md
│   ├── phase2_simulator.md
│   ├── phase3_observability.md
│   ├── references.md
│   └── decisions/
├── src/r2dn_dc_motor/data/
│   └── sequences.py
├── src/r2dn_dc_motor/numerics/
│   └── rk4.py
├── src/r2dn_dc_motor/plants/
│   └── electrothermal.py
├── src/r2dn_dc_motor/validation/
│   ├── phase2.py
│   └── phase3.py
├── src/r2dn_dc_motor/models/
│   ├── checkpoint.py
│   ├── r2dn_adapter.py
│   └── temperature_probe.py
├── src/r2dn_dc_motor/
│   ├── phase1_spec.py
│   ├── phase2_spec.py
│   ├── phase3_spec.py
│   ├── spec.py
│   ├── validate_phase0.py
│   ├── validate_phase1.py
│   ├── validate_phase2.py
│   └── validate_phase3.py
└── tests/
```

Konfiguracije u `configs/` su mašinski čitljivi izvori istine. Testovi odbijaju
curenje temperature, drift interfejsa i fizičkog domena, teacher forcing u
slobodnom rollout-u, nedovoljno fin RK4 korak i checkpoint napravljen drugim
upstream commitom.

Gate 1 dodatno odbija sample-level podelu pilot podataka, temperaturu u
ulaznim obeležjima, termički signal uporediv sa numeričkom greškom i
identifikabilnost koja postoji samo u trenutnom uzorku bez istorije.

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
