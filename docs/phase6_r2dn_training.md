# Faza 6 — implementacija i trening R2DN world modela

## Ugovor podataka

R2DN koristi isključivo normalizovani Phase-4 `model_view`:

\[
r_k=[i_k,\omega_k,u_k],\qquad
\hat y_{k+1}=[\hat i_{k+1},\hat\omega_{k+1}].
\]

Temperatura, stvarno opterećenje, referenca brzine i komandovani napon nisu
polja trening batch-a. Normalizacija je nepromenjena train-only normalizacija
iz Phase-4 dataseta.

## Burn-in i loss

Finalni profil koristi 250 koraka, odnosno 250 ms merene istorije. Posle
burn-in intervala autoregresivni deo više ne dobija stvarne buduće opservacije.

\[
\mathcal L =
\lambda_1\mathcal L_{\mathrm{one}}
+\lambda_H\mathcal L_{\mathrm{rollout}}
+\lambda_{\mathrm{rec}}\mathcal L_{\mathrm{burn}}.
\]

Curriculum povećava slobodni rollout sa jednog na 10, 100 i 1000 koraka.
I pilot koristi međufazu od 10 koraka pre horizonta 100, da prelazak sa
isključivo teacher-forced one-step loss-a ne proizvede numerički overflow u
autoregresivnom gradijentu. Struja i brzina imaju jednaku težinu tek nakon
train-only normalizacije.

## Pilot i ponovljivost

Pilot poredi latentne dimenzije 4, 6 i 8. Izabrana arhitektura zatim se trenira
od početka sa seed-ovima 17, 29 i 43. Najbolji finalni checkpoint bira se
isključivo po validation free-rollout NRMSE na horizontu od 5000 koraka.
Svi latentni kandidati dele iste pilot-validation prozore, a sva tri finalna
seed-a dele isti drugi skup validation prozora.

ID i OOD skup se ne čitaju ni za fit ni za izbor checkpoint-a.

## Checkpoint

```text
checkpoints/phase6/r2dn-v1/
├── manifest.json
├── parameters.msgpack
├── normalization.npz
└── training_history.json
```

Manifest sadrži fingerprint dataseta, pinned upstream commit, arhitekturu,
seed, burn-in, selection horizont, contractivity margin i SHA-256 svakog
pratećeg fajla. Loader odbija checkpoint drugog dataseta ili izmenjen sadržaj.

## Lokalni finalni trening

Za NVIDIA GPU koristi se zaključani JAX 0.5.3 CUDA 12 plugin. Instalacija sa
`phase6-cuda12` povlači CUDA/cuDNN biblioteke kao Python pakete; potreban je
kompatibilan NVIDIA drajver, ali zasebna lokalna CUDA Toolkit instalacija nije
potrebna.

```bash
python -m pip install -e ".[dev,phase6,phase6-cuda12]"

python -m r2dn_dc_motor.validate_phase6 \
  --runtime-only \
  --require-cuda

python -m r2dn_dc_motor.validate_phase6 \
  --train \
  --profile final \
  --require-cuda \
  --dataset data/phase4-full-v1 \
  --checkpoint-dir checkpoints/phase6/r2dn-v1 \
  --output-dir results/phase6
```

Preflight mora da prikaže `backend: gpu` i `CUDA active: yes`. Opcija
`--require-cuda` prekida program pre učitavanja dataseta ako JAX pokuša tihi
CPU fallback. Aktivni backend, modeli uređaja i verzije CUDA plugin/PJRT paketa
čuvaju se u `training_history.json`.

Postojeći checkpoint može se namerno zameniti samo dodatkom
`--overwrite-checkpoint`.

Kratka provera celog mehanizma koristi `--profile ci` i Phase-4 `ci` dataset.
Ona radi samo nekoliko gradient update-a, pa njena greška nije naučni rezultat.

## Granica faze

Phase 6 je završena kada finalni trening za sva tri seed-a daje konačne loss,
gradijent i validation rollout vrednosti, kontraktivni sertifikat ostaje
pozitivan, predikcije ostaju unutar zaključanog konzervativnog normalizovanog
opsega i checkpoint prolazi svih osam zaštita.

Poređenje sa ISO-CAL na istim burn-in/prognostičkim prozorima, cold/hot i
ID/OOD analiza i konačni uslov za prelazak na RL pripadaju Fazi 7.
