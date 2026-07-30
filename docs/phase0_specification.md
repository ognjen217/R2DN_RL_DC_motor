# Faza 0 — zaključana eksperimentalna postavka

## 1. Istraživačko pitanje

Da li policy trenirana u R2DN world modelu ima manji transfer gap pri
zero-shot prenosu na kompletan elektrotermalni DC motor od iste policy
arhitekture trenirane u kalibrisanom izotermskom ODE modelu?

Glavna hipoteza je:

\[
\Delta J_{\mathrm{PPO-R2DN}}
<
\Delta J_{\mathrm{PPO-ISO-CAL}}.
\]

## 2. Referentni sistem

Kompletno stanje je:

\[
x_k =
\begin{bmatrix}
i_k & \omega_k & T_k
\end{bmatrix}^{\mathsf T},
\]

gde su \(i\) struja armature, \(\omega\) ugaona brzina, a \(T\) temperatura
namotaja.

Merljivi izlaz je:

\[
y_k =
\begin{bmatrix}
i_k & \omega_k
\end{bmatrix}^{\mathsf T}.
\]

Upravljanje je napon armature:

\[
u_k \in [-48,48]\ \mathrm{V}.
\]

Cilj upravljanja je praćenje reference brzine:

\[
\omega_k \rightarrow \omega_k^{\mathrm{ref}}.
\]

## 3. Informaciona ograničenja

Temperatura:

- postoji u FULL simulatoru;
- utiče na otpor namotaja;
- ne ulazi u world-model interfejs;
- ne ulazi u policy opservaciju;
- koristi se samo za dijagnostiku i evaluaciju.

World model dobija:

\[
[i_k,\omega_k,u_k],
\]

a policy dobija:

\[
[i_k,\omega_k,\omega_k^{\mathrm{ref}},u_{k-1}].
\]

U osnovnom eksperimentu temperatura je jedino skriveno dinamičko stanje.
Moment opterećenja je poznat i konstantan, a temperatura okoline konstantna.
Nepoznato opterećenje, šum i parametarske promene pripadaju kasnijim
stress-testovima.

## 4. Zaključani modeli i regulatori

| Oznaka | Uloga |
|---|---|
| FULL | Kompletni elektrotermalni ODE model i završno postrojenje |
| ISO-NOM | Nominalni izotermski ODE model sa RK4 |
| ISO-CAL | Globalno kalibrisani izotermski ODE model sa RK4 |
| R2DN | Kontraktivni rekurentni world model |
| PI | Klasični kontrolni benchmark |
| PPO-FULL | Oracle praktična referenca bez dostupne temperature |

GRU je opciona ablacija i ne pripada obaveznom osnovnom eksperimentu.

## 5. Vremenska reprezentacija

FULL, ISO-NOM i ISO-CAL su kontinualni ODE modeli integrisani RK4 metodom.
Kontrolni period je \(1\ \mathrm{ms}\), a početni unutrašnji korak integratora
\(0.1\ \mathrm{ms}\). Tačnost koraka biće proverena konvergencionim testom u
fazi implementacije simulatora.

R2DN ostaje diskretni rekurentni state-space model, u skladu sa referentnim
radom i zvaničnom implementacijom:

\[
x_{t+1}=f_\theta(x_t,u_t), \qquad
y_t=h_\theta(x_t,u_t).
\]

Nije potrebno da neuralni model koristi RK4 samo zato što ga koriste fizički
baseline modeli. Svi backend-i dele isti spoljašnji kontrolni period.

## 6. Zaključani domeni

| Signal | Minimum | Maksimum |
|---|---:|---:|
| Napon armature | \(-48\ \mathrm{V}\) | \(48\ \mathrm{V}\) |
| Struja armature | \(-25\ \mathrm{A}\) | \(25\ \mathrm{A}\) |
| Ugaona brzina | \(-500\ \mathrm{rad/s}\) | \(500\ \mathrm{rad/s}\) |
| Referenca brzine | \(-400\ \mathrm{rad/s}\) | \(400\ \mathrm{rad/s}\) |
| Temperatura namotaja | \(20^\circ\mathrm{C}\) | \(120^\circ\mathrm{C}\) |

Ovo su bezbednosni i eksperimentalni domeni, a ne tvrdnja da će svi uzorci
trening skupa ravnomerno pokriti ceo opseg.

## 7. Primarne metrike

Za kvalitet world modela primarna metrika je višekoračni NRMSE. Za upravljanje
je primarna metrika IAE brzine na FULL postrojenju:

\[
\operatorname{IAE}
=
\sum_k
\left|\omega_k-\omega_k^{\mathrm{ref}}\right|T_s.
\]

Transfer gap je:

\[
\Delta J_\pi
=
\left|
J_{\mathrm{world}}-J_{\mathrm{FULL}}
\right|.
\]

Sekundarne metrike obuhvataju one-step grešku, maksimalnu grešku,
nestabilne rollout-e, ISE, prekoračenje, vreme smirivanja, struju, upravljački
napor, saturaciju, temperaturu, uspešnost epizode i vreme izvršavanja.

## 8. Kriterijum završetka Faze 0

Faza je završena kada:

- `configs/phase0.toml` prolazi validator;
- test potvrđuje da temperatura ne curi u world model ili policy;
- katalog modela i primarne metrike odgovaraju osnovnoj hipotezi;
- fizički i R2DN backend imaju zaključanu vremensku reprezentaciju;
- zvanična R2DN implementacija je identifikovana i vezana za commit;
- CI može automatski da izvrši iste provere.

