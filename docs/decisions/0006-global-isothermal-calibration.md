# ADR 0006: Jedna globalna, temperature-free ISO-CAL kalibracija

## Status

Prihvaćeno u Fazi 5.

## Odluka

`ISO-CAL` ima istu dvostanjsku izotermsku strukturu kao `ISO-NOM`. Kalibrišu
se samo efektivni otpor \(R_{\mathrm{eff}}\) i viskozno trenje \(b\), jednim
globalnim fitom nad svim celim train trajektorijama.

Kalibracija pristupa isključivo Phase-4 `model_view` vrednostima
\((i,\omega,u)\). Ne čita temperaturu, stvarno opterećenje, referencu brzine
ni komandovani napon. Opterećenje u modelu ostaje fiksirano na nominalnoj
vrednosti iz Faze 2.

## Razlozi

- efektivni otpor daje fizičkom baseline-u najbolju poštenu statičku
  aproksimaciju termički promenljivog otpora;
- jedna globalna vrednost ne može indirektno kodirati termički režim test
  epizode;
- zadržavanje iste strukture razdvaja korist kalibracije parametara od koristi
  rekurentne memorije;
- neprosleđivanje stvarnog opterećenja čuva isti world-model ulaz i stvarnu
  OOD proveru jačeg opterećenja.

## Posledice

`ISO-CAL` može biti bolji od `ISO-NOM`, ali i dalje ne može prilagoditi otpor
trenutnoj skrivenoj temperaturi. Upravo je ta preostala greška cilj budućeg
poređenja sa R2DN modelom.
