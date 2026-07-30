# Faza 5 — fizički baseline modeli

Faza 5 uvodi dva namerno nepotpuna izotermska modela sa istim spoljašnjim
world-model interfejsom:

\[
y_k=[i_k,\omega_k],\qquad u_k=V_k,\qquad
\hat y_{k+1}=f(y_k,u_k).
\]

Oba modela koriste

\[
L\dot i=u-Ri-k_e\omega,
\]

\[
J\dot\omega=k_ti-b\omega-T_{L,\mathrm{nom}},
\]

i nemaju termičko stanje.

## ISO-NOM

`ISO-NOM` direktno koristi nominalne netermičke parametre iz
`configs/phase2.toml`. Njegov efektivni otpor jednak je referentnom otporu
\(R_0\). Model ne koristi temperaturu ni tokom inicijalizacije ni tokom
rollout-a.

## ISO-CAL

`ISO-CAL` zadržava potpuno istu strukturu kao `ISO-NOM`, ali procenjuje jedan
globalni efektivni otpor \(R_{\mathrm{eff}}\) i jedan globalni koeficijent
viskoznog trenja \(b\). Parametri \(L,J,k_e,k_t\) i nominalno opterećenje ostaju
fiksirani.

Kalibracija koristi svaki prelaz iz svake cele `train` trajektorije. Jedino
dozvoljeno sučelje je Phase-4 `model_view`:

```text
armature_current_a, angular_speed_rad_s, armature_voltage_v
```

Temperatura, stvarno opterećenje, referenca brzine i komandovani napon ne
ulaze u fit. Posebno je važno da se stvarno opterećenje ne prosleđuje
baseline-u: model uvek pretpostavlja \(T_{L,\mathrm{nom}}\), pa promena
opterećenja ostaje stvarna OOD osa.

## Postupak kalibracije

Za svaki prelaz formiraju se midpoint aproksimacije

\[
i_{k+1/2}=\frac{i_k+i_{k+1}}{2},\qquad
\omega_{k+1/2}=\frac{\omega_k+\omega_{k+1}}{2},
\]

i konačne razlike. Zatim se akumuliraju dve skalarne least-squares normalne
jednačine:

\[
R_{\mathrm{eff}}=
\frac{\sum i_{k+1/2}
\left(u_k-k_e\omega_{k+1/2}-L\frac{\Delta i_k}{T_s}\right)}
{\sum i_{k+1/2}^2},
\]

\[
b=
\frac{\sum \omega_{k+1/2}
\left(k_ti_{k+1/2}-T_{L,\mathrm{nom}}
-J\frac{\Delta\omega_k}{T_s}\right)}
{\sum \omega_{k+1/2}^2}.
\]

Akumulacija je streaming: trajektorije se učitavaju jedna po jedna, pa finalni
dataset ne mora ceo biti u RAM-u. Dobijene vrednosti ograničavaju se unapred
zaključanim fizičkim granicama iz `configs/phase5.toml`.

## Checkpoint i zabrana curenja

Checkpoint `checkpoints/phase5/iso_cal.json` sadrži:

- jedan globalni skup parametara;
- fingerprint Phase-4 dataseta;
- kompletan spisak korišćenih train trajektorija;
- dozvoljene i zabranjene feature-e;
- sufficient statistics i broj prelaza;
- zaključani metod i protokol izbora.

Loader odbija checkpoint drugog dataseta, nepotpun train skup, duplirane
trajektorije i checkpoint koji tvrdi da je koristio temperaturu ili stvarno
opterećenje.

## Evaluacija

Oba modela se bez dodatnog podešavanja mere na `validation`, `id_test` i
`ood_test` podelama. Izveštaj sadrži:

- teacher-forced one-step NRMSE;
- rollout NRMSE za 0,1 s, 1 s i 10 s;
- dugi rollout posebno za hladne i zagrejane početne režime;
- brzinu izvršavanja u prelazima u sekundi.

Greške struje i brzine dele se standardnim devijacijama iz Phase-4
train-only normalizacije. Temperatura se čita samo da bi se evaluacione
trajektorije označile kao hladne ili zagrejane; ne ulazi u predikciju.

## Pokretanje na finalnom datasetu

```bash
python -m pip install -e ".[dev,phase5]"

python -m r2dn_dc_motor.validate_phase5 \
  --fit \
  --dataset data/phase4-full-v1 \
  --checkpoint checkpoints/phase5/iso_cal.json \
  --output-dir results/phase5
```

Za namerno ponavljanje kalibracije i zamenu checkpoint-a dodaje se
`--overwrite-checkpoint`. Bez `--fit`, komanda učitava postojeći checkpoint i
samo ponavlja evaluaciju.
