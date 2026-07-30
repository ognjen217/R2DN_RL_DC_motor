# Faza 1 — Projektovanje i provera R2DN interfejsa

## Cilj

Faza 1 zaključava način na koji će zvanični R2DN kod biti korišćen kao world
model. U ovoj fazi se ne implementiraju motor, dataset, trening niti RL.

Autoritativni mašinski čitljiv dokument je `configs/phase1.toml`, a njegovu
usklađenost sa Fazom 0 proverava:

```bash
python -m r2dn_dc_motor.validate_phase1
```

## Mapiranje zahteva na zvanični kod

| Zahtev projekta | Zvanični `ContractingR2DN` | Projektni sloj |
|---|---|---|
| Eksplicitni kontrolni ulaz | proizvoljan vektor `inputs` | ulaz je `[i_k, omega_k, u_k]` |
| Trajno latentno stanje | `state`/`carry` | stanje se prenosi između svih koraka |
| Latentna dimenzija > 2 | `state_size` je konfigurabilan | minimum 3, pilot 4 |
| Batch sekvence | `simulate_sequence`, raspored `(T,B,F)` | isti raspored je zaključan |
| Burn-in | nema namenski metod | adapter koristi merene `[y_k,u_k]` |
| Slobodni rollout | nema autoregresivni wrapper | predikcija `y_hat_k` ulazi u sledeći korak |
| Kontraktivnost | direktna parametrizacija | test reziduala uslova (20) |

## Tačna vremenska semantika

Jedan trening prelaz je:

\[
r_k=[i_k,\omega_k,u_k],
\qquad
\hat y_{k+1}=\operatorname{R2DN}(z_k,r_k),
\]

gde je:

\[
y_k=[i_k,\omega_k].
\]

Batch sadrži:

- `observations`: `(T + 1, B, 2)`;
- `controls`: `(T, B, 1)`;
- `regressors`: `(T, B, 3)`;
- `targets`: `(T, B, 2)`.

Raspored osa je uvek `time, batch, feature`.

## Burn-in i slobodni rollout

Latentno stanje se na resetu postavlja na nulu, a zatim se kroz burn-in
obrađuje merena istorija:

\[
(y_0,u_0),\ldots,(y_{B-1},u_{B-1}).
\]

Posle burn-in intervala poslednja merena opservacija \(y_B\) služi samo kao
početno seme rollout-a. Za \(k\ge B\), buduća stvarna opservacija se ne vraća
modelu:

\[
\hat y_{k+1}
=
\operatorname{R2DN}(z_k,[\hat y_k,u_k]).
\]

Dužina burn-in intervala nije proizvoljno zaključana u Fazi 1. Biće izvedena
nakon analize električne, mehaničke i termičke vremenske konstante u Fazi 2.

## Precizno značenje kontraktivnosti

R2DN rad definiše kontraktivnost za dve latentne putanje sa različitim početnim
stanjima i istom spoljašnjom ulaznom sekvencom. Dodavanje kontrolnog signala
\(u_k\) ne menja ovu garanciju ako obe putanje dobijaju isti signal.

Upstream direktna parametrizacija gradi matricu:

\[
H
=
X^\top X+\epsilon I
+
\begin{bmatrix}
C_1^\top C_1 & 0\\
0 & \mathcal B_1\mathcal B_1^\top
\end{bmatrix},
\]

pa je rezidual uslova (20) strogo pozitivno definitan. Integracioni test računa
njegovu najmanju sopstvenu vrednost.

U našem autoregresivnom rollout-u predikovani izlaz postaje deo sledećeg ulaza.
Zato dve putanje više nemaju nužno identičan kompletan regressor. Formalna
upstream garancija se ne proširuje automatski na tu spoljašnju povratnu spregu.
Njena stabilnost mora se meriti dugim numeričkim rollout-ima.

## Zaštita od curenja temperature

`FullTrajectoryBatch` čuva redosled `[i, omega, T]`, ali
`model_view()` eksplicitno izbacuje temperaturu i vraća samo `[i, omega]`.
Objekat `ModelSequenceBatch` nema polje temperature. Testovi dodatno odbijaju
temperaturu u regressor-u i target-u konfiguracije.

## Checkpoint format

Jedan budući checkpoint je direktorijum sa:

```text
checkpoint/
├── manifest.json
├── parameters.msgpack
└── normalization.npz
```

Manifest zaključava:

- upstream commit;
- redosled opservacija i upravljanja;
- latentnu dimenziju i širine mreže;
- seed;
- kontrolni period;
- verziju šeme.

Checkpoint sa drugim upstream commitom ili promenjenim redosledom signala
programski se odbija.

## Kriterijum završetka Faze 1

Faza 1 je završena kada:

1. oba validatora daju `PASS`;
2. obavezni testovi prolaze bez opcionog backend-a;
3. pinned upstream backend može da se inicijalizuje;
4. latentno stanje se prenosi kroz burn-in;
5. free rollout ne koristi buduće stvarne opservacije i ostaje numerički konačan;
6. rezidual uslova (20) ima pozitivnu najmanju sopstvenu vrednost;
7. lint je čist.

Ovi kriterijumi ne tvrde da je neistrenirani model tačan. Tačnost predikcije
pripada Fazama 6 i 7.
