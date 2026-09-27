# Detailed Failure Analysis Report: STCQA

- **Source File:** `experiments/predictions/stcqa_test_preds.json`
- **Total Samples:** 1063
- **Total Missed in Top 10:** 975 (91.72%)

## Accuracy & Failure Rate Breakdown

| Category | Count | Hits@1 | Hits@10 | Missed in Top 10 (%) |
| :--- | :---: | :---: | :---: | :---: |
| Overall | 1063 | 1.51% | 8.28% | 91.72% |
| DTC | 28 | 0.00% | 3.57% | 96.43% |
| STC | 1035 | 1.55% | 8.41% | 91.59% |
| DDC | 330 | 2.12% | 9.39% | 90.61% |
| SDC | 378 | 2.12% | 12.43% | 87.57% |
| DC | 355 | 0.28% | 2.82% | 97.18% |

## Selected Failure Cases for Qualitative Inspection

### Case #1 (Sample ID: 0)
- **Question:** `What <isCitizenOf> <Alan_Tate> after it <hasNeighbor> <Saint_Barthélemy> within 137 miles of <China> ?`
- **Paraphrased:** *"What country is Alan Tate a citizen of that is located within 137 miles of China and has Saint Barthélemy as a neighbor?"*
- **Constraints:** Temporal: `STC` | Spatial: `DC`
- **Gold Answer(s):** **<Belgium>**
- **Gold Rank:** `Not in Top 10`
- **Top Predictions:**
  1. `<Colchester_United_F.C.>`
  2. `<Nastassja_Kinski>`
  3. `<Tokyo>`
  4. `<Charlton_Athletic_F.C.>`
  5. `<University_of_Pennsylvania>`

### Case #2 (Sample ID: 1)
- **Question:** `Later than the termination of <Quebec_City>, give me the team did <Djamel_Abdoun> join southeast of <Dundee> .`
- **Paraphrased:** *"Which team did Djamel Abdoun join after leaving Quebec City, specifically in the southeastern area of Dundee?"*
- **Constraints:** Temporal: `STC` | Spatial: `DDC`
- **Gold Answer(s):** **<Manchester_City_F.C.>**
- **Gold Rank:** `Not in Top 10`
- **Top Predictions:**
  1. `<Immanuel_Kant>`
  2. `<Fyodor_Dostoyevsky>`
  3. `<Johann_Wolfgang_von_Goethe>`
  4. `<Mila_Kunis>`
  5. `<Leo_Tolstoy>`

### Case #3 (Sample ID: 2)
- **Question:** `What <isCitizenOf> <Gioachino_Rossini> before it <hasNeighbor> <Vatican_City> within 321 miles of <United_Kingdom> ?`
- **Paraphrased:** *"What country is Gioachino Rossini a citizen of that is located within 321 miles of the United Kingdom and neighbors Vatican City?"*
- **Constraints:** Temporal: `STC` | Spatial: `DC`
- **Gold Answer(s):** **<Italy>**
- **Gold Rank:** `Not in Top 10`
- **Top Predictions:**
  1. `<Rotherham_United_F.C.>`
  2. `<Saint_Barthélemy>`
  3. `<Rhine>`
  4. `<Russia>`
  5. `<Germany>`

### Case #4 (Sample ID: 3)
- **Question:** `Posterior to the cessation of <Walt_Disney_Animation_Studios>, <Pinto_Colvig> leave which college northeast of <Samoa> .`
- **Paraphrased:** *"After Walt Disney Animation Studios came to an end, which college did Pinto Colvig depart from, located to the northeast of Samoa?"*
- **Constraints:** Temporal: `STC` | Spatial: `DDC`
- **Gold Answer(s):** **<Oregon_State_University>**
- **Gold Rank:** `Not in Top 10`
- **Top Predictions:**
  1. `<United_Kingdom>`
  2. `<Czech_Republic>`
  3. `<Switzerland>`
  4. `<Wigan_Athletic_F.C.>`
  5. `<Motherwell_F.C.>`

### Case #5 (Sample ID: 4)
- **Question:** `<Khari_Stephenson> joined which country west of  the <Aalesunds_FK> she played for later than the dissolution of <AS_Nancy> ?`
- **Paraphrased:** *"After the dissolution of AS Nancy, which country west of Aalesunds FK did Khari Stephenson join?"*
- **Constraints:** Temporal: `STC` | Spatial: `SDC`
- **Gold Answer(s):** **<Jamaica>**
- **Gold Rank:** `Not in Top 10`
- **Top Predictions:**
  1. `<Italy>`
  2. `<California>`
  3. `<Aston_Villa_F.C.>`
  4. `<Iran>`
  5. `<Austria>`

### Case #6 (Sample ID: 5)
- **Question:** `<Libya> establish transaction relationship with which countries northwest of it after the termination of <Highland_Park,_Illinois> ?`
- **Paraphrased:** *"After the termination of Highland Park, Illinois, which countries northwest of Libya did it establish a transactional relationship with?"*
- **Constraints:** Temporal: `STC` | Spatial: `DDC`
- **Gold Answer(s):** **<France>, <Germany>, <Netherlands>, <Italy>, <Tunisia>, <Spain>**
- **Gold Rank:** `Not in Top 10`
- **Top Predictions:**
  1. `<Cardiff_City_F.C.>`
  2. `<Northampton_Town_F.C.>`
  3. `<Bristol_Rovers_F.C.>`
  4. `<Immanuel_Kant>`
  5. `<Rotherham_United_F.C.>`

### Case #7 (Sample ID: 6)
- **Question:** `What <hasNeighbor> <Mali> before it <hasNeighbor> <Nigeria> within 570 miles of <Nigeria> ?`
- **Paraphrased:** *"Which country borders Mali before it borders Nigeria within 570 miles of Nigeria?"*
- **Constraints:** Temporal: `STC` | Spatial: `DC`
- **Gold Answer(s):** **<Niger>**
- **Gold Rank:** `Not in Top 10`
- **Top Predictions:**
  1. `<Bristol_City_F.C.>`
  2. `<California>`
  3. `<Barnsley_F.C.>`
  4. `<William_Faulkner>`
  5. `<China>`

### Case #8 (Sample ID: 7)
- **Question:** `Founded posterior to <Pratt_Institute>, which countries east of <South_Africa> ?`
- **Paraphrased:** *"Which countries are located to the east of South Africa and were established after Pratt Institute?"*
- **Constraints:** Temporal: `STC` | Spatial: `SDC`
- **Gold Answer(s):** **<Zimbabwe>, <Mozambique>, <Lesotho>**
- **Gold Rank:** `Not in Top 10`
- **Top Predictions:**
  1. `<Australia>`
  2. `<Switzerland>`
  3. `<Portugal>`
  4. `<Southend_United_F.C.>`
  5. `<Poland>`

### Case #9 (Sample ID: 8)
- **Question:** `What <isCitizenOf> <Immanuel_Kant> before it <hasNeighbor> <Finland> within 1824 miles of <Japan> ?`
- **Paraphrased:** *"What country was Immanuel Kant a citizen of before it had Finland as a neighbor within 2,581 miles of Japan?"*
- **Constraints:** Temporal: `STC` | Spatial: `DC`
- **Gold Answer(s):** **<Russia>**
- **Gold Rank:** `Not in Top 10`
- **Top Predictions:**
  1. `<Charles_Darwin>`
  2. `<Joaquin_Phoenix>`
  3. `<Mila_Kunis>`
  4. `<Josip_Broz_Tito>`
  5. `<Johann_Wolfgang_von_Goethe>`

### Case #10 (Sample ID: 9)
- **Question:** `After the termination of <Park_County,_Wyoming>, can you give me the team east of <Princeton,_New_Jersey> did <Ross_Turnbull> join  ?`
- **Paraphrased:** *"Can you tell me which team Ross Turnbull joined after Park County, Wyoming was terminated, specifically the one located east of Princeton, New Jersey?"*
- **Constraints:** Temporal: `STC` | Spatial: `SDC`
- **Gold Answer(s):** **<Darlington_F.C.>**
- **Gold Rank:** `Not in Top 10`
- **Top Predictions:**
  1. `<Frederick_the_Great>`
  2. `<Samuel_Goldwyn>`
  3. `<Strasbourg>`
  4. `<Mila_Kunis>`
  5. `<Fyodor_Dostoyevsky>`

### Case #11 (Sample ID: 10)
- **Question:** `Later than <Platoon_(film)>, which countries southeast of <Burkina_Faso> and establish transaction relationship with it ?`
- **Paraphrased:** *"Which countries to the southeast of Burkina Faso establish transaction relationships with, after the release of the film Platoon?"*
- **Constraints:** Temporal: `STC` | Spatial: `DDC`
- **Gold Answer(s):** **<Togo>, <Ghana>, <Benin>**
- **Gold Rank:** `Not in Top 10`
- **Top Predictions:**
  1. `<United_Kingdom>`
  2. `<Immanuel_Kant>`
  3. `<Spain>`
  4. `<Johann_Wolfgang_von_Goethe>`
  5. `<Rotherham_United_F.C.>`

### Case #12 (Sample ID: 11)
- **Question:** `What <isCitizenOf> <Norman_Taurog> before it <hasNeighbor> <Estonia> within 3330 miles of <Belarus> ?`
- **Paraphrased:** *"Before Estonia is within 3330 miles of Belarus, what is Norman Taurog a citizen of?"*
- **Constraints:** Temporal: `STC` | Spatial: `DC`
- **Gold Answer(s):** **<Russia>**
- **Gold Rank:** `Not in Top 10`
- **Top Predictions:**
  1. `<Cardiff_City_F.C.>`
  2. `<Rangers_F.C.>`
  3. `<Southend_United_F.C.>`
  4. `<Blackpool_F.C.>`
  5. `<Middlesbrough_F.C.>`

### Case #13 (Sample ID: 12)
- **Question:** `Can you list all countries northwest of <New_Zealand> and have business deal with it after the termination of <Boston_College> ?`
- **Paraphrased:** *"After Boston College ends, would you be able to provide a list of countries located northwest of New Zealand that you could potentially conduct business with?"*
- **Constraints:** Temporal: `STC` | Spatial: `DDC`
- **Gold Answer(s):** **<United_States>, <Germany>, <Japan>, <South_Korea>, <Malaysia>, <China>, <Australia>**
- **Gold Rank:** `Not in Top 10`
- **Top Predictions:**
  1. `<Northampton_Town_F.C.>`
  2. `<Bristol_Rovers_F.C.>`
  3. `<Hollywood>`
  4. `<Om_Shanti_Om>`
  5. `<Walt_Disney>`

### Case #14 (Sample ID: 13)
- **Question:** `Later than the cessation of <Houston_Rockets>, <Austria> have transaction with which countries east of it ?`
- **Paraphrased:** *"Which countries east of Austria does it engage in transactions with after the Houston Rockets stopped?"*
- **Constraints:** Temporal: `STC` | Spatial: `SDC`
- **Gold Answer(s):** **<Czech_Republic>, <Slovakia>**
- **Gold Rank:** `Not in Top 10`
- **Top Predictions:**
  1. `<Cardiff_City_F.C.>`
  2. `<Russia>`
  3. `<Alan_Fettis>`
  4. `<Alan_Kennedy>`
  5. `<House_of_Flying_Daggers>`

### Case #15 (Sample ID: 14)
- **Question:** `Posterior to the dissolution of <Highland_Park,_Illinois>, can you list all countries west of <Syria> and establish transaction relationship with it ?`
- **Paraphrased:** *"After the dissolution of Highland Park, Illinois, can you provide a list of all countries located west of Syria and establish any transactional relationships with them?"*
- **Constraints:** Temporal: `STC` | Spatial: `SDC`
- **Gold Answer(s):** **<France>, <Libya>, <Turkey>**
- **Gold Rank:** `Not in Top 10`
- **Top Predictions:**
  1. `<RC_Lens>`
  2. `<Leon_Trotsky>`
  3. `<Tarzana,_Los_Angeles>`
  4. `<Bill_Whittaker_(footballer)>`
  5. `<Tianjin>`

### Case #16 (Sample ID: 15)
- **Question:** `After the dissolution of <Gloucester_County,_New_Jersey>, which countries south of <Russia> and have business deal with it ?`
- **Paraphrased:** *"Which countries, located south of Russia, have business dealings with Gloucester County, New Jersey, following its dissolution?"*
- **Constraints:** Temporal: `STC` | Spatial: `SDC`
- **Gold Answer(s):** **<United_States>, <Germany>, <Belarus>, <Netherlands>, <Japan>, <Italy>, <Turkey>, <China>**
- **Gold Rank:** `Not in Top 10`
- **Top Predictions:**
  1. `<Blackpool_F.C.>`
  2. `<Los_Angeles>`
  3. `<Wimbledon_F.C.>`
  4. `<Cardiff_City_F.C.>`
  5. `<Southend_United_F.C.>`

### Case #17 (Sample ID: 16)
- **Question:** `After the termination of <Rotherham_United_F.C.>, <Saudi_Arabia> establish transaction relationship with which countries north of it ?`
- **Paraphrased:** *"Which countries located north of Saudi Arabia did establish a transactional relationship after the termination of Rotherham United F.C.?"*
- **Constraints:** Temporal: `STC` | Spatial: `SDC`
- **Gold Answer(s):** **<United_States>, <Germany>, <Japan>, <South_Korea>, <United_Kingdom>, <China>**
- **Gold Rank:** `Not in Top 10`
- **Top Predictions:**
  1. `<Mexico>`
  2. `<Northampton_Town_F.C.>`
  3. `<Poland>`
  4. `<Nigeria>`
  5. `<Stoke_City_F.C.>`

### Case #18 (Sample ID: 17)
- **Question:** `Name the club southwest of <Hamburg> did <Jamar_Loza> affiliate with later than the cessation of <Central_African_Republic> .`
- **Paraphrased:** *"What is the name of the club that Jamar_Loza became affiliated with after the Central_African_Republic ceased to exist?"*
- **Constraints:** Temporal: `STC` | Spatial: `DDC`
- **Gold Answer(s):** **<Coventry_City_F.C.>**
- **Gold Rank:** `Not in Top 10`
- **Top Predictions:**
  1. `<Austria>`
  2. `<Fyodor_Dostoyevsky>`
  3. `<Yo-Yo_Ma>`
  4. `<Spain>`
  5. `<Chow_Yun-fat>`

### Case #19 (Sample ID: 18)
- **Question:** `What <isCitizenOf> <Michael_Balcon> before it <hasNeighbor> <Estonia> within 144 miles of <Finland> ?`
- **Paraphrased:** *"Before Estonia is within 144 miles of Finland, what is Michael Balcon a citizen of?"*
- **Constraints:** Temporal: `STC` | Spatial: `DC`
- **Gold Answer(s):** **<Latvia>**
- **Gold Rank:** `Not in Top 10`
- **Top Predictions:**
  1. `<Bristol_Rovers_F.C.>`
  2. `<Kuwait>`
  3. `<United_Kingdom>`
  4. `<Wimbledon_F.C.>`
  5. `<Singapore>`

### Case #20 (Sample ID: 19)
- **Question:** `<Indonesia> have transaction with which countries north of it later than the termination of <Bellevue,_Washington> ?`
- **Paraphrased:** *"Which countries north of Indonesia have had transactions with it after the termination of Bellevue, Washington?"*
- **Constraints:** Temporal: `STC` | Spatial: `SDC`
- **Gold Answer(s):** **<Thailand>, <Singapore>, <United_States>, <Netherlands>, <Japan>, <Malaysia>, <India>, <South_Korea>, <China>**
- **Gold Rank:** `Not in Top 10`
- **Top Predictions:**
  1. `<Russia>`
  2. `<Hibernian_F.C.>`
  3. `<Charles_de_Gaulle>`
  4. `<Niccolò_Machiavelli>`
  5. `<Fulham_F.C.>`
