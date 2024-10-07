from faker import Faker
from random import choice
import csv

fake = Faker('en_US')

def student_info(rows):
  student_response = []

  class_response = ['SS1', 'SS2', 'SS3']
  gender = ['Female', 'Male']
  age = ['12-14', '15-17', '18 and above']
  school = ['Public', 'Private']
  father_edu_background = ['No formal education', 'Primary','Secondary','University']
  mother_edu_background = ['No formal education', 'Primary','Secondary','University']
  father_job = ['Civil servant','Engineer','Civil Servant','Farmer','Tailoring' ]
  mother_job = ['Trader','Teacher','House Wife','Trader','Housewife']
  household_income = ['Below ₦30,000','₦30,000 – ₦100,000','50,000 - 100,000','₦100,000 – ₦300,000','Above ₦300,000']
  school_has_library = ['Yes','No']
  library_visit = ['Daily','Weekly','Monthly','Rarely','Never']
  access_to_computer_lab = ['Yes','No']
  extra_mural_class = ['Yes','No']
  extra_curricular_activities = ['Yes', 'No']
  extra_curricular_participation = ['Sports','Debate/Quiz clubs','Science/Math clubs','None']
  hours_on_curricular_activities = ['1–3 hours','4–6 hours','7 hours or more','None']
  english_score = ['39 below','40 - 44','45 - 49','50 - 54','55 - 59','60 - 64','65 - 69','70 - 75','80 - 100']
  maths_score = ['39 below','40 - 44','45 - 49','50 - 54','55 - 59','60 - 64','65 - 69','70 - 75','80 - 100']
  hours_spent_studying = ['Less than 1 hour','1–2 hours','3–4 hours','5 hours or more']
  difficulty_understand_maths_eng_sci = ['Yes','No']
  subject_found_most_difficult = ['Mathematics','English Language','Physics/Chemistry/Biology','Other']
  confidence_in_passing_exam = ['Very confident','Somewhat confident','Not confident']
  prepare_exam_with_online_resources = ['Yes','No']
  challenge_preparing_for_exam = ['Lack of focus', 'time management','The fear of the unknown',
                                  'Lack of proper preparation','Lack of funds', 'lack of standard social amenity'
                                  ]

  for _ in range(rows):
    data = {
      'class': choice(class_response),
      'gender': choice(gender),
      'age': choice(age),
      'school': choice(school),
      'father_edu_background': choice(father_edu_background),
      'mother_edu_background': choice(mother_edu_background),
      'father_job': choice(father_job),
      'mother_job': choice(mother_job),
      'household_income': choice(household_income),
      'school_has_library': choice(school_has_library),
      'library_visit': choice(library_visit),
      'access_to_computer_lab': choice(access_to_computer_lab),
      'extra_mural_class': choice(extra_mural_class),
      'extra_curricular_activities': choice(extra_curricular_activities),
      'extra_curricular_participation': choice(extra_curricular_participation),
      'hours_on_curricular_activities': choice(hours_on_curricular_activities),
      'english_score': choice(english_score),
      'maths_score': choice(maths_score),
      'hours_spent_studying': choice(hours_spent_studying),
      'difficulty_understand_maths_eng_sci': choice(difficulty_understand_maths_eng_sci),
      'subject_found_most_difficult': choice(subject_found_most_difficult),
      'confidence_in_passing_exam': choice(confidence_in_passing_exam),
      'prepare_exam_with_online_resources': choice(prepare_exam_with_online_resources),
      'challenge_preparing_for_exam': choice(challenge_preparing_for_exam)



    }

    student_response.append(data)
  
  return student_response


def save_as_csv(data, filename):
  with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=data[0].keys())
    writer.writeheader()
    writer.writerows(data)

student_info_ = student_info(1800)
save_as_csv(student_info_, './student_survey.csv')
