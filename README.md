# R2DN world model za upravljanje DC motorom

Eksperimentalni kod za ispitivanje da li kontraktivni rekurentni model R2DN
omogućava bolji transfer RL regulatora od nepotpunog izotermskog fizičkog
modela kada DC motor ima skrivenu elektrotermalnu dinamiku.

## Trenutni status

Implementirana je **Faza 0 — zaključavanje eksperimentalne postavke**:

- formalno su definisani stanje, opservacije, upravljanje i skrivena veličina;
- zaključan je glavni skup modela i regulatora;
- definisani su domeni signala i primarne metrike;
- temperatura je programski isključena iz world-model i policy interfejsa;
- zvanični R2DN kod i verzija rada evidentirani su kao referenca;
- testovi sprečavaju nenameran drift specifikacije.

Simulator motora, dataset, R2DN trening i RL još nisu implementirani.

## Brza provera

Potreban je Python 3.11 ili noviji.

```bash
python -m pip install -e ".[dev]"
python -m r2dn_dc_motor.validate_phase0
pytest
```

Očekivani rezultat validatora je:

```text
PHASE 0: PASS
```

## Struktura

```text
.
├── configs/phase0.toml
├── docs/
│   ├── phase0_specification.md
│   ├── references.md
│   └── decisions/
├── src/r2dn_dc_motor/
│   ├── spec.py
│   └── validate_phase0.py
└── tests/test_phase0_spec.py
```

`configs/phase0.toml` je mašinski čitljiv izvor istine. Izmene koje menjaju
eksperimentalnu hipotezu, dostupne opservacije, glavne modele ili primarne
metrike moraju biti namerne, verzionisane i praćene izmenom specifikacije.

## R2DN referenca

Koristiće se zvanična JAX implementacija iz
[`nic-barbara/R2DN`](https://github.com/nic-barbara/R2DN), vezana za konkretan
commit u `configs/phase0.toml`. Kod nije kopiran u ovaj repozitorijum u Fazi 0.

R2DN je u referentnom radu diskretni rekurentni state-space model. Zbog toga će
R2DN backend koristiti diskretne korake, dok će fizički ODE modeli koristiti
RK4 između kontrolnih trenutaka.

        