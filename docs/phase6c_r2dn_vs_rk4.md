# Faza 6C — R2DN naspram FULL/RK4 na 1000 s

## Cilj

Faza 6B je pokazala da izabrani R2DN ostaje konačan i fizički ograničen tokom
milion autoregresivnih koraka. Taj rezultat sam za sebe ne pokazuje koliko je
R2DN brz niti koliko se njegova trajektorija slaže sa punim elektrotermalnim
modelom. Faza 6C zato pokreće R2DN i kanonski `FULL/RK4` nad istim budućim
naponom tokom 1000 s.

Poređenje koristi:

- kontrolni period od 1 ms, odnosno tačno \(10^6\) kontrolnih koraka;
- zaključani RK4 korak od 0,1 ms, odnosno deset RK4 podkoraka po kontrolnom
  koraku i ukupno \(10^7\) RK4 podkoraka;
- isti Phase-6B stress anchor, početnu struju i brzinu;
- stvarnu skrivenu temperaturu anchor-a samo za inicijalizaciju FULL modela;
- isti budući napon i podrazumevani konstantni moment opterećenja;
- 250 merenih koraka za inicijalizaciju R2DN latentnog stanja.

Podrazumevani `multisine` scenario je izabran jer ne svodi poređenje na jednu
ravnotežnu tačku. CLI prihvata i svih ostalih sedam zaključanih Phase-6B
scenarija.

## Metrike

JSON izveštaj sadrži:

- hladno R2DN vreme, koje uključuje JIT kompilaciju;
- zagrejano R2DN vreme;
- vreme kanonskog NumPy/Python FULL/RK4 simulatora;
- simulirane sekunde po jednoj zidnoj sekundi;
- izmereni odnos `RK4 vreme / R2DN vreme`;
- RMSE struje u A i brzine u rad/s;
- NRMSE po izlazu i kombinovani NRMSE, korišćenjem train-only standardnih
  devijacija iz Phase-4 dataseta;
- kumulativne greške na 1, 10, 100 i 1000 s;
- maksimalnu i završnu grešku;
- tačne identitete dataseta, checkpoint-a, seed-a i stress anchor-a.

Obe merene putanje materijalizuju kompletnu izlaznu trajektoriju. R2DN warm
vreme uključuje burn-in, JAX chunk dispatch, izvršavanje i prenos izlazne
putanje sa uređaja na host.

## Granica tvrdnje o brzini

Glavni odnos vremena poredi postojeće implementacije: FULL/RK4 radi kroz
NumPy/Python na CPU-u, dok R2DN radi kroz JAX na uređaju navedenom u JSON-u.
Zato je to merodavno end-to-end ubrzanje trenutnog eksperimentalnog sistema,
ali nije hardverski nezavisna tvrdnja da je R2DN algoritam toliko puta brži od
svake moguće optimizovane RK4 implementacije.

## Pokretanje

Iz korena repozitorijuma:

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m r2dn_dc_motor.compare_r2dn_rk4 \
  --require-cuda \
  --dataset data/phase4-full-v1 \
  --checkpoint-dir checkpoints/phase6b/r2dn-v2 \
  --phase6b-report results/phase6b/phase6b_latent_and_stability.json \
  --scenario multisine \
  --duration-s 1000 \
  --split validation \
  --anchor-index 0 \
  --chunk-steps 10000 \
  --output-dir results/phase6c
```

Dobijaju se:

```text
results/phase6c/phase6c_r2dn_vs_rk4_1000s.json
results/phase6c/phase6c_r2dn_vs_rk4_1000s.png
```

RK4 deo sadrži deset miliona podkoraka i može trajati znatno duže od R2DN
dela. Napredak se zato ispisuje pre početka svake metode, a završni rezultat se
upisuje tek kada su obe putanje kompletne.
