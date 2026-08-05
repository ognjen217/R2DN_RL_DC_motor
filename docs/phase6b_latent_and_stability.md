# Faza 6B — proširena latentna pretraga i autoregresivni stress test

## Motivacija

Faza 6 je završila pilot na latentnim dimenzijama 4, 6 i 8, pri čemu je najbolji
kandidat bio upravo 8, odnosno gornja granica početne pretrage. Zbog toga Faza
6 nije dovoljna da se tvrdi da je latent 8 optimalan. Takođe, pozitivan
kontraktivni margin zvaničnog R2DN-a garantuje zaboravljanje latentnog početnog
stanja za istu spoljašnju sekvencu regresora, ali ne sertifikuje proširenu
autoregresivnu petlju u kojoj se \(\hat y_k\) vraća u sledeći regresor.

Faza 6B rešava oba pitanja pre zajedničkog Phase-7 poređenja.

## Zaključana pretraga latentne dimenzije

Osnovni katalog je

\[
n_z\in\{4,6,8,10,12,16\}.
\]

Svaki kandidat koristi:

- iste Phase-6 pilot stage-ove \(1\rightarrow10\rightarrow100\);
- ista tri seed-a: 1701, 2701 i 3701;
- iste validation prozore (`seed=61001`);
- isti validation horizont od 500 koraka;
- isti train-only normalizer i temperature-free ulaz \((i,\omega,u)\).

Za svaki latent računa se medijana tri validation free-rollout NRMSE rezultata.
Prvo se nalazi najbolja medijana. Ako je neki manji latent najviše 3% lošiji od
nje, bira se manji model. Time jedan srećan seed ili zanemarljivo poboljšanje
većeg modela ne određuju arhitekturu.

Latent 24 se dodaje samo ako medijana latenta 16 bude strogo više od 5% bolja
od medijane latenta 12. Ovo je unapred definisano pravilo proširenja pretrage,
a ne odluka doneta posle gledanja rezultata.

Izabrani latent se trenira istim finalnim Phase-6 curriculumom i seed-ovima
17, 29 i 43. Finalni checkpoint i dalje bira najmanji 5000-koračni validation
free-rollout NRMSE. Ako ponovo pobedi latent 8, već validirani Phase-6
checkpoint može se ponovo upotrebiti: njegov finalni trening, tri seed-a i
selection kriterijum identični su Phase-6B finalnom delu.

Finalni trening koristi finite-update guard iz Faze 6: autoregresivni minibatch
sa nekonačnim loss-om, gradijentom ili kandidatom optimizer stanja ne menja
parametre. Sledeći deterministički prozor zamenjuje odbijeni batch, a broj
retry-ja ulazi u istoriju. Više od 5% odbijenih batch-eva u jednoj curriculum
fazi i dalje prekida run, jer bi to ukazivalo na sistemski, a ne izolovani
numerički problem.

Svaki pilot i finalni run čuva se odmah po završetku u resumable cache-u
vezanom za dataset fingerprint i SHA-256 celog Phase-6/6B protokola.

## Held-out akumulacija greške

Nakon izbora modela, greška se meri na fiksnim prozorima iz validation, ID i OOD
podela na horizontima 1 s, 5 s i 10 s. Meri se:

- ukupni, strujni i brzinski free-rollout NRMSE;
- maksimalna apsolutna normalizovana predikcija;
- konačnost celog rollout-a.

Ovi rezultati su isključivo post-selection dijagnostika. Ne smeju promeniti
latent, seed ili checkpoint.

## Sintetički dugi stress test

Model se posle validnog 250 ms burn-in intervala pušta autoregresivno tokom
10 s, 100 s i 1000 s. Za svaki od validation, ID i OOD režima koriste se četiri
fiksna početna burn-in prozora i osam primenjenih naponskih sekvenci:

| Scenario | Primenjeni napon |
|---|---|
| `zero_voltage` | \(0\ \mathrm V\) |
| `constant_positive` | \(+10\ \mathrm V\) |
| `constant_negative` | \(-10\ \mathrm V\) |
| `prbs` | PRBS, \(\pm18\ \mathrm V\), hold 50 ms |
| `sine` | sinus, amplituda \(15\ \mathrm V\), 1 Hz |
| `multisine` | tri komponente, zbir amplituda \(15\ \mathrm V\) |
| `positive_voltage_limit` | \(+20\ \mathrm V\) |
| `negative_voltage_limit` | \(-20\ \mathrm V\) |

Svi naponi ostaju unutar Phase-4 sigurnog domena. Rollout se izvršava u
chunk-ovima od 10.000 koraka, tako da nije potrebno čuvati milion latentnih
stanja u GPU memoriji.

Za svaki milestone beleže se:

- prvi `NaN`/`Inf`, ako postoji;
- prvi izlazak struje ili brzine iz Phase-2 fizičkog domena;
- maksimumi \(|i|\), \(|\omega|\), \(\|z\|\) i normalizovane predikcije;
- elektromagnetna i kinetička energija

  \[
  E_{\mathrm{em},k}
  =\frac12 L\hat i_k^2+\frac12 J\hat\omega_k^2;
  \]

- kumulativni električni rad \(\sum_k u_k\hat i_k\Delta t\);
- odnos RMS/amplitude između dve polovine poslednjih 10% rollout-a.

## Osetljivost proširene autoregresivne petlje

Svaki stress rollout istovremeno izvršava baznu i perturbovanu putanju pod
istim naponom. Posle zajedničkog burn-in-a dodaje se perturbacija norme
\(10^{-3}\) normalizovanom izlazu i latentnom stanju. Beleže se maksimalno,
krajnje i tail-RMS rastojanje izlaza. Ovo numerički proverava da li mala razlika
\((z_0,\hat y_0)\) raste u spoljnoj autoregresivnoj petlji.

## Kriterijum prolaza i ograničenje tvrdnje

Na najdužem horizontu svih 24 kombinacija split/scenario moraju:

- ostati konačne;
- ostati u fizičkim granicama struje i brzine;
- ostati ispod konzervativne granice 25 u normalizovanim koordinatama;
- zadržati latentnu normu ispod \(10^4\) bez trajnog rasta u repu;
- pri nultom naponu ne pokazivati trajan rast energije u repu;
- ne pokazivati trajan rast male početne perturbacije.

Energijska provera je fizička dijagnostika, ne formalni dokaz pasivnosti. R2DN
ne predviđa temperaturu i ne prima stvarno opterećenje, pa se iz njegovih izlaza
ne može rekonstruisati kompletan elektrotermomehanički bilans.

I uspešan milion-koračni test je konačan numerički dokaz za zaključane režime,
a ne matematički dokaz za \(k\rightarrow\infty\) i svaki mogući ulaz.

## Pokretanje

```bash
python -m pip install -e ".[dev,phase6,phase6-cuda12]"

CUDA_VISIBLE_DEVICES=0 \
python -m r2dn_dc_motor.validate_phase6b \
  --runtime-only \
  --require-cuda

CUDA_VISIBLE_DEVICES=0 \
python -m r2dn_dc_motor.validate_phase6b \
  --search \
  --stress \
  --profile final \
  --require-cuda \
  --dataset data/phase4-full-v1 \
  --phase6-checkpoint checkpoints/phase6/r2dn-v1 \
  --cache-dir checkpoints/phase6b/run-cache-v1 \
  --checkpoint-dir checkpoints/phase6b/r2dn-v2 \
  --output-dir results/phase6b
```

Ponovno pokretanje koristi završene run cache-ove. `--overwrite-cache` i
`--overwrite-checkpoint` koriste se samo za namerno ponovno treniranje i
zamenu postojećeg rezultata. Ako je selected checkpoint već uspešno sačuvan,
a prekinut je samo dugi stress test, komanda se ponavlja samo sa `--stress`,
bez `--search`.

Glavni izlazi su:

```text
checkpoints/phase6b/r2dn-v2/
├── manifest.json
├── parameters.msgpack
├── normalization.npz
└── study_history.json

results/phase6b/
├── phase6b_latent_and_stability.json
└── phase6b_latent_and_stability.png
```
