# Poređenje skrivenog termičkog uticaja na horizontima 1–1000 s

## Istraživačko pitanje

Eksperiment proverava da li kontraktivni data-driven model može, bez direktnog
merenja temperature, bolje da aproksimira posledice skrivene elektrotermalne
dinamike od dva nepotpuna izotermska fizička modela.

Sva tri kandidata koriste isti runtime interfejs:

```text
observations = [armature_current_a, angular_speed_rad_s]
control      = [armature_voltage_v]
```

Temperatura se ne prosleđuje ni R2DN-u, ni ISO-NOM-u, ni ISO-CAL-u. `FULL/RK4`
je ground-truth baseline: temperatura je njegovo interno stanje, a otpor namotaja
se menja prema stvarnoj implementiranoj temperaturnoj zavisnosti.

## Zaključani modeli

- **R2DN:** Phase-6E pobednik, latent 16, seed 43, checkpoint
  `checkpoints/phase6e/r2dn-v1`. Phase 6F je potvrdio da taj postojeći model
  ostaje pobednik optimizer ablacije.
- **ISO-NOM:** nominalni izotermski model, bez kalibracije na testu.
- **ISO-CAL:** jedan globalni Phase-5 fit napravljen samo na train trajektorijama;
  kalibrišu se isključivo efektivni otpor i viskozno trenje. Temperatura i stvarni
  moment opterećenja nisu korišćeni u fitu.
- **FULL/RK4:** puni elektrotermalni motor, deset RK4 podkoraka od 0,1 ms po
  kontrolnom periodu od 1 ms.

Svi rollout-i polaze iz iste struje i brzine, koriste isti primenjeni napon i
nikada se ne reinicijalizuju između horizonta. Metrike na 1, 10, 100 i 1000 s su
kumulativne metrike prefiksa jedne iste milion-koračne putanje.

## Pokretanje

Iz korena kompletnog repozitorijuma:

```bash
source .venv/bin/activate
python -m pip install -e ".[dev,phase6,phase6-cuda12]"

CUDA_VISIBLE_DEVICES=0 \
python -m r2dn_dc_motor.compare_hidden_thermal_models \
  --require-cuda \
  --dataset data/phase4-full-v1 \
  --checkpoint-dir checkpoints/phase6e/r2dn-v1 \
  --iso-cal-checkpoint checkpoints/phase5/iso_cal.json \
  --phase6b-report results/phase6b/phase6b_latent_and_stability.json \
  --scenario multisine \
  --duration-s 1000 \
  --split validation \
  --anchor-index 0 \
  --chunk-steps 10000 \
  --output-dir results/hidden_thermal_comparison
```

Dobijaju se:

```text
results/hidden_thermal_comparison/hidden_thermal_comparison_1000s.json
results/hidden_thermal_comparison/hidden_thermal_comparison_1000s.png
```

JSON sadrži NRMSE struje, brzine i kombinovani NRMSE za svaki kandidat na sva
četiri horizonta, rangiranje po kombinovanom NRMSE-u, fizičke RMSE metrike,
maksimalne i završne greške, vremena izvršavanja, identitet checkpoint-a i
audit kojim se potvrđuje da temperatura nije korišćena kao ulaz.

PNG sadrži:

1. struju sva tri kandidata i FULL/RK4 baseline-a;
2. brzinu sva tri kandidata i FULL/RK4 baseline-a;
3. kombinovani NRMSE kroz horizonte 1/10/100/1000 s;
4. skrivenu temperaturu FULL modela i zajednički napon.

## Tumačenje

Ako R2DN ima manji NRMSE od ISO-NOM i ISO-CAL, eksperiment podržava tvrdnju da
je iz istorije merljivih signala naučio deo efekta skrivene temperature koji
izotermski modeli sa jednim konstantnim otporom ne mogu da reprodukuju. Rezultat
ne dokazuje da R2DN eksplicitno rekonstruiše temperaturu, niti da može da
generalizuje na proizvoljne termičke režime van trening domena.
