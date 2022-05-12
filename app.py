import numpy as np
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
import plotly.express as px
from plotnine import ggplot, aes, geom_point, geom_smooth, labs

with st.echo(code_location='below'):
    st.write("# Проект по визуализации данных по Формуле-1.")

    @st.cache(allow_output_mutation=True)
    def get_data(data):
        return pd.read_csv(data)

    circuits = get_data("circuits.csv")
    constructor_results = get_data("constructor_results.csv")
    constructor_standings = get_data("ConstructorStandings.csv")
    constructors = get_data("constructors.csv")
    driver_standings = get_data("driver_standings.csv")
    drivers = get_data("drivers.csv")
    lap_times = get_data("lap_times.csv")
    pit_stops = get_data("pit_stops.csv")
    qualifying = get_data("qualifying.csv")
    races = get_data("races.csv")
    results = get_data("results.csv")
    seasons = get_data("seasons.csv")
    sprint_results = get_data("sprint_results.csv")


    drivers['fullname'] = drivers['forename'] + ' ' + drivers['surname']

    """Этот проект, как следует из названия, посвящён анализу и визуализации различных данных, связанных с Формулой-1.
    Формула-1 - это гоночный вид спорта, где спортсмены борются за победу, выступая на разных трассах. 
    Поэтому основные объекты для анализа и визуализации данных по Формуле-1 - это выступления гонщиков и команд в 
    разных гонках, а также во всём сезоне суммарно."""
    """Проект можно разделить на несколько частей и я предлагаю приступить к первой из них!"""

    st.write("## Часть 1: Анализ данных по отдельным гонщикам и странам")

    """Начнём с анализа гонщиков Формулы-1."""
    """Ниже вы можете вписать в окошко полное имя (имя и фамилию на английском языке) и посмотреть краткую информацию 
    об этом пилоте. Помимо кратких сведений о нём (его номере, национальности и дате рождения), можно посмотреть, 
    сколько побед, вторых и третьих мест было с карьере этого гонщика и какую часть от всех гонок они составляют.
    Если вы не знаете никаких гонщиков Формулы-1, то можете выбрать кого-нибудь из списка, расположенного ниже 
    или найти в интернете:\n
    Fernando Alonso, Michael Schumacher, Lewis Hamilton, Daniil Kvyat, Sebastian Vettel, Ayrton Senna"""

    name = st.text_input("Введите полное имя гонщика:", key='name')

    if name:
        results_by_driver = results.merge(drivers, left_on='driverId', right_on='driverId')[
                                lambda x: x['fullname'] == name]
        first_places = results_by_driver[lambda x: x['positionOrder'] == 1]['raceId'].count()
        second_places = results_by_driver[lambda x: x['positionOrder'] == 2]['raceId'].count()
        third_places = results_by_driver[lambda x: x['positionOrder'] == 3]['raceId'].count()
        finishes = results_by_driver['raceId'].count()
        other_places = finishes - first_places - second_places - third_places
        driver_results = pd.DataFrame({'Places': ['1st place', '2nd place', '3rd place', '4th or lower'],
                                       'Number': [first_places, second_places, third_places, other_places]})
        if not results_by_driver.empty:
            (drivers[drivers['fullname'] == name].reset_index()
            [['forename', 'surname', 'code', 'number', 'dob', 'nationality']])

            fig = px.pie(driver_results, values='Number', names='Places', title=f'The career performance of {name}:',
                         color='Places',
                         color_discrete_map={'1st place': 'gold',
                                             '2nd place': 'gray',
                                             '3rd place': 'saddlebrown',
                                             '4th or lower': 'whitesmoke'})
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig)
        else:
            """Такого гонщика нет в базе данных. Попробуйте снова!"""

    """Далее вы можете выбрать какую-нибудь страну из списка и посмотреть, на каких автодромах этой страны проводились 
    Гран-При Формулы-1. Ниже Вы увидите визуализацию: круговую диаграмму, показывающую какую часть от всех гонок, 
    проведённых в этой стране принял каждый из автодромов"""

    country = st.selectbox("Choose the country", circuits["country"].unique())

    circuits_in_country = circuits[lambda x: x['country'] == country].reset_index()[['name','location','country']]
    circuits_in_country

    races_by_circuits = (races.merge(circuits, left_on='circuitId', right_on='circuitId')
                         [lambda x: x['country'] == country])
    number_of_races_by_circuits = races_by_circuits.groupby('name_y')['raceId'].count()
    number_of_races_by_circuits_df = (pd.DataFrame({'Track': number_of_races_by_circuits.index,
                                                    'Number of races': number_of_races_by_circuits}))

    fig = px.pie(number_of_races_by_circuits_df, values='Number of races', names='Track',
                 title=f'The number of races on the tracks located in {country}:',
                 color='Track')
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig)

    st.write("## Часть 2: Анализ данных по разным странам")

    """Теперь перейдём к следующему разделу. Ранее мы рассмотрели отдельно гонщиков и страны, а теперь попробуем 
    рассмотреть их вместе."""

    """Ниже вы можете выбрать какую-либо страну и увидеть, какие гонщики этой национальности становились победителями 
    Гран-При Формулы-1, а также сравнить их между собой по количеству побед. 
    Обратите внимание, что у некоторых стран нет своих гонщиков-победителей (например, у Японии и России), 
    а в каких-то странах (например, Испания и Польша) победителем становился только один гонщик."""

    nationality = st.selectbox("Выберите страну (национальность):", drivers["nationality"].unique())


    @st.cache
    def victories_by_drivers():
        return results[lambda x: x['positionOrder'] == 1].merge(drivers, left_on='driverId', right_on='driverId')

    victories_by_drivers = victories_by_drivers()

    @st.cache
    def nationality_info(nationality):
        number_of_victories = (victories_by_drivers[lambda x: x['nationality'] == nationality]
                               .groupby("fullname")['raceId'].count())
        return (pd.DataFrame({'Driver': number_of_victories.index, 'Number of victories': number_of_victories})
            .sort_values('Number of victories', ascending=False).reset_index()[['Driver', 'Number of victories']])

    total_victories = nationality_info(nationality)

    if not total_victories.empty:
        fig, ax = plt.subplots()
        chart = sns.barplot(y=total_victories['Driver'], x=total_victories['Number of victories'] , ax=ax,
                            palette='crest')
        chart.bar_label(chart.containers[0], fontsize=8.5, color='black')
        chart.set_title(f'The {nationality} drivers with victories')
        st.pyplot(fig)
    else:
        """У этой страны пока нет гонщиков победителей. Выберете другую!"""


    """Как вы могли увидеть, количество победителей и побед в разных странах сильно отличается. 
    Мне кажется, было бы интересно сравнить разные страны (национальности) друг с другом. Чтобы сделать это, 
    выберите в следующем окошке несколько стран."""
    nationalities =  st.multiselect("Выберите страны (национальности):", drivers["nationality"].unique())
    number_of_victories_by_nations = (victories_by_drivers[lambda x: x['nationality'].isin(nationalities)]
                                      .groupby("nationality")['raceId'].count())
    total_victories_by_nations = (pd.DataFrame({'Nationality': number_of_victories_by_nations.reset_index()['nationality'],
                                               'Number of victories': number_of_victories_by_nations.reset_index()['raceId']})
                                  .sort_values('Number of victories', ascending=False))
    for element in nationalities:
        if not total_victories_by_nations['Nationality'].isin([element]).any():
            total_victories_by_nations = (total_victories_by_nations
                                          .append({'Nationality': element, 'Number of victories': 0}, ignore_index=True))
    if not total_victories_by_nations.empty:
        fig, ax = plt.subplots()
        chart = sns.barplot(y=total_victories_by_nations['Nationality'], x=total_victories_by_nations['Number of victories'], ax=ax)
        chart.bar_label(chart.containers[0], fontsize=8.5, color='black')
        chart.set_title(f'The comparison of the nationalities')
        st.pyplot(fig)
    else:
        """Никакие национальности не были выбраны."""

    st.write("## Часть 3: Немного статистики и аналитики")

    """Для начала сравним гонщиков между собой и найдём лучших из них по нескольким параметрам.
    Я предлагаю найти топ-20 гонщиков по следующим показателям: количество проведённых гонок, количество подиумов 
    (1-ое, 2-ое и 3-ье места) и количество побед. Ниже Вы можете выбрать одby из этих параметров и посмотреть, 
    какие гонщики являются лучшими по каждому из них"""

    choice = st.radio('Выберете:', ['Количество проведённых гонок', 'Количество подиумов', 'Количество побед'])

    @st.cache
    def total_results():
        return results.merge(drivers, left_on='driverId', right_on='driverId')

    total_results = total_results()

    @st.cache
    def races_dr():
        all_drivers_races = total_results.groupby('fullname')['raceId'].count()
        return (pd.DataFrame({'Driver': all_drivers_races.index, 'Number of races': all_drivers_races})
                    .sort_values('Number of races', ascending=False).iloc[0:20])

    @st.cache
    def victories():
        all_drivers_victories = total_results[lambda x: x['positionOrder'] == 1].groupby('fullname')['raceId'].count()
        return (pd.DataFrame({'Driver': all_drivers_victories.index, 'Number of victories': all_drivers_victories})
                    .sort_values('Number of victories', ascending=False).iloc[0:20])

    @st.cache
    def podiums():
        all_drivers_victories = total_results[lambda x: x['positionOrder'] == 1].groupby('fullname')['raceId'].count()
        all_drivers_2nd = total_results[lambda x: x['positionOrder'] == 2].groupby('fullname')['raceId'].count()
        all_drivers_3rd = total_results[lambda x: x['positionOrder'] == 3].groupby('fullname')['raceId'].count()
        all_drivers_podiums = all_drivers_victories + all_drivers_2nd + all_drivers_3rd
        return (pd.DataFrame({'Driver': all_drivers_podiums.index, 'Number of podiums': all_drivers_podiums})
                    .sort_values('Number of podiums', ascending=False).iloc[0:20])

    all_drivers_races_df = races_dr()
    all_drivers_victories_df = victories()
    all_drivers_podiums_df = podiums()

    if choice == 'Количество проведённых гонок':
        fig, ax = plt.subplots()
        chart1 = sns.barplot(y=all_drivers_races_df['Driver'], x=all_drivers_races_df['Number of races'],
                             ax=ax, palette="viridis")
        chart1.bar_label(chart1.containers[0], fontsize=8.5, color='black')
        chart1.set_title('The drivers with the highest number of races')
        st.pyplot(fig)

    if choice == 'Количество подиумов':
        fig, ax = plt.subplots()
        chart2 = sns.barplot(y=all_drivers_podiums_df['Driver'], x=all_drivers_podiums_df['Number of podiums'],
                             ax=ax, palette='flare')
        chart2.bar_label(chart2.containers[0], fontsize=8.5, color='black')
        chart2.set_title('The drivers with the highest nimber of podiums')
        st.pyplot(fig)

    if choice == 'Количество побед':
        fig, ax = plt.subplots()
        chart3 = sns.barplot(y=all_drivers_victories_df['Driver'], x=all_drivers_victories_df['Number of victories'],
                             ax=ax, palette='rocket')
        chart3.bar_label(chart3.containers[0], fontsize=8.5, color='black')
        chart3.set_title('The drivers with the highest number of victories')
        st.pyplot(fig)

    """Как Вы могли заметить, некоторые гонщики есть во всех трёх "топах", но также много гонщиков появляются 
    лишь в одном или двух из них. В связи с этим я предлагаю посмотреть, как связано количество проведённых Гран-При, 
    количество побед и призовых мест."""

    """Для этого рассмотрим следующие графики:"""

    @st.cache
    def data_for_graphs():
        results_by_drivers = results.merge(drivers, left_on='driverId', right_on='driverId')
        first_places1 = results_by_drivers[lambda x: x['positionOrder'] == 1].groupby('fullname')['raceId'].count()
        second_places1 = results_by_drivers[lambda x: x['positionOrder'] == 2].groupby('fullname')['raceId'].count()
        third_places1 = results_by_drivers[lambda x: x['positionOrder'] == 3].groupby('fullname')['raceId'].count()
        finishes = results_by_drivers.groupby('fullname')['raceId'].count()
        podiums1 = first_places1 + second_places1 + third_places1
        total_driver_results = pd.DataFrame()
        total_driver_results['finishes'] = finishes
        total_driver_results['victories'] = first_places1
        total_driver_results['podiums'] = podiums1
        return total_driver_results.fillna(0)

    total_driver_results = data_for_graphs()

    plot1 = (ggplot(total_driver_results, aes(x='finishes', y='victories'))
             + labs(x="Races", y="Victories", title="The correlation between the numbers of victories and races",
                    color="Number of races")
             + geom_point(aes(color='finishes')) + geom_smooth(method='lm')
             )
    st.pyplot(ggplot.draw(plot1))

    plot2 = (ggplot(total_driver_results, aes(x='finishes', y='podiums'))
             + geom_point(aes(color='finishes')) + geom_smooth(method='lm') +
             labs(x="Races", y="Podiums", title="The correlation between the numbers of podiums and races",
                    color="Number of races"))
    st.pyplot(ggplot.draw(plot2))

    """Давайте теперь разберёмся, что изображено на этих графиках."""
    """На них отмечено множество точек. 
    Эти точки отображают информацию обо всех гонщиках: количество проведённых гонок и побед/подиумов.
    А чёрная линия - график функции регресии. С помощью неё можно сделать предположение, сколько в среднем 
    побед/подиумов сможет добиться гонщик проведя какое-то опредёлённое количество гонок. 
    Например, за 100 гонок выигрывают около 6-7 гонок и приехжают на подиум приблизительно в 20 гонках."""
    """Посмотрим на ещё один график:"""

    plot3 = (ggplot(total_driver_results[lambda x: x['podiums'] > 0], aes(x='podiums', y='victories'))
             + geom_point(aes(color='finishes')) + geom_smooth(method='lm') +
             labs(x="Podiums", y="Victories", title="The correlation between the numbers of victories and podiums",
                  color="Number of races")
             )
    st.pyplot(ggplot.draw(plot3))

    """Этот график показывает взаимосвязь между количеством подиумов и побед гонщиков, а наклон чёрной линии 
    показывает, какой в среднем процент подиумов приходится на победы."""

    st.write("## Часть 4: Динамика")

    """Далее я предлагаю добавить немного динамики!"""

    """Следующая визуализация показывает, как менялось количество побед в гонках у разных команд (конструкторов) 
    с течением времени. Я предлагаю посмотреть на 15 лучших команд (по суммарному количеству побед) и на то, когда они 
    добивались своих побед и в каких масштабах."""

    """Обратите внимание, что эта визуализация анимирована: Вы можете выбрать какой-то конкретный год или запустить 
    анимацию с 1950-ого года или любого другого момента."""

    @st.cache
    def animation1():
        data = (results[lambda x: x['positionOrder'] == 1]
                .merge(constructors, left_on='constructorId', right_on='constructorId')
                .merge(races[['year', 'raceId']], left_on='raceId', right_on='raceId')[
                    ['name', 'positionOrder', 'year']].sort_values('year'))
        data1 = data.pivot_table(values='positionOrder', index='name', columns='year', aggfunc='count').fillna(0)
        data2 = np.cumsum(data1, axis=1).unstack().reset_index()
        data2['victories'] = data2[0]
        top_teams = data2.sort_values('victories', ascending=False)['name'].unique()[0:15]
        data3 = data2[lambda x: x['name'].isin(top_teams)]
        figure = px.bar(data3, x='victories', y="name", color="name",
                        animation_frame="year", labels={"victories": "Victories", "name": "Teams", "year": 'Year'},
                title="The dynamics of victories for top 15 teams", width=700, height=700)

        return figure

    st.plotly_chart(animation1())

    st.write("## Часть 5: Анализ гоночного этапа")

    """Обычный гоночный этап в Формуле-1 состоит из сеансов свободной практики, когда гонщики тренируются, 
    из квалификации, когда гонщики показывают время, которое определит, на какой позиции они будут стартовать
    и самой гонки, за которую выдаются призы и начисляются очки. Свободные практики для нас не представляют особого 
    интереса, поэтому далее мы рассмотрим только квалификацию и гонку."""

    st.write("### Квалификация")

    """Квалификация в свою очередь состоит из трёх сегментов (такой регламент используется с 2006 года).
    В первом сегменте участвуют все гонщики, но в следующий этап проходит лишь часть (обычно отсеиваются 5-6 гонщиков 
    с худшими врменами). Во втором сегменте происходит то же самое: 5-6 самых медленных гонщиков из оставшихся не 
    проходят дальше. В третьем сегменте решается судьба поул-позиции (права стартовать с первой позиции в гонке).
    За это право сражаются оставшиеся 8-10 самых быстрых гонщиков."""

    """Далее я предлагаю выбрать любой Гран-При, который прошёл с 2006 года по 2021 год. 
    Для этого сначала выберете год, а затем один из доступных этапов. 
    После этого Вы можете выюрать одного из гонщиков, участвоваших в этой квалификации и посмотреть, как он выступил"""

    """На этой визуализации каждая точка - отдельный пилот и его лучшее время в сегменте. Наведя курсор мышки на точку, 
    Вы можете посмотреть, какому пилоту соответсвует эта точка и какое время он показал в данном сегменте. 
    Чёрными точками отмечены результаты ранее выбранного пилота."""

    """Вы можете выбрать поочереди разных пилотов и выбирать разные этапы, чтобы посмотреть, кто обычно хорошо выступает 
    в квалификациях, а кто хуже. Также можно сравнить, как менялись времена гонщика в разных сегментах одной квалификации."""

    a = st.slider('Choose the year:', 2006, 2021)
    races_in_this_year = races[lambda x: x['year'] == a]
    grand_prix = st.selectbox('Choose the Gran Prix:', races_in_this_year['name'].unique())

    @st.cache(allow_output_mutation=True)
    def df_for_q(a, grand_prix):
        df1 = qualifying.merge(races[lambda x: x['year'] == a], left_on='raceId', right_on='raceId')
        df = df1[lambda x: x['name'] == grand_prix].merge(drivers, left_on='driverId', right_on='driverId')
        ### FROM (частично) https://question-it.com/questions/1146283/pandas-preobrazovanie-vremeni-v-sekundy-dlja-vseh-znachenij-v-stolbtse
        df['q1'] = (df[df['q1'].str[0].isin(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])]
                    ['q1'].map(lambda x: sum(x * float(t) for x, t in zip([60.0, 1.0], x.split(':')))))
        df['q2'] = (df[df['q2'].str[0].isin(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])]
                    ['q2'].map(lambda x: sum(x * float(t) for x, t in zip([60.0, 1.0], x.split(':')))))
        df['q3'] = (df[df['q3'].str[0].isin(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])]
                    ['q3'].map(lambda x: sum(x * float(t) for x, t in zip([60.0, 1.0], x.split(':')))))
        ### END FROM
        df['type1'] = 'Q1'
        df['type2'] = 'Q2'
        df['type3'] = 'Q3'
        return df

    df = df_for_q(a, grand_prix)

    chosen_driver = st.selectbox('Choose the driver:', df['fullname'].unique())

    df2 = df[lambda x: x['fullname'] == chosen_driver]
    df3 = pd.DataFrame(
        {'time': [df2.iloc[0]['q1'], df2.iloc[0]['q2'], df2.iloc[0]['q3']], 'session': ['Q1', 'Q2', 'Q3']})

    @st.cache
    def figure1():
        return (alt.Chart(df).mark_point(size=50, filled=True)
            .encode(alt.X('q1', scale=alt.Scale(zero=False), axis=alt.Axis(title='Lap time')),
                    alt.Y('type1', scale=alt.Scale(zero=False), axis=alt.Axis(title='Session')),
                    color=alt.Color('fullname', legend=alt.Legend(title='The drivers')),
                    tooltip = [alt.Tooltip('fullname'), alt.Tooltip('q1')])
            .properties(height=700, width=700, title=f'The qualification of {grand_prix} {a}').interactive())

    @st.cache
    def figure2():
        return (alt.Chart(df).mark_point(size=50, filled=True)
         .encode(alt.X('q2', scale=alt.Scale(zero=False), axis=alt.Axis(title='Lap time')),
                 alt.Y('type2', scale=alt.Scale(zero=False), axis=alt.Axis(title='Session')),
                 color=alt.Color('fullname', legend=alt.Legend(title='The drivers')),
                 tooltip=[alt.Tooltip('fullname'), alt.Tooltip('q2')])
         .properties(height=700, width=700, title=f'The qualification of {grand_prix} {a}').interactive())

    @st.cache
    def figure3():
        return (alt.Chart(df).mark_point(size=50, filled=True)
            .encode(alt.X('q3', scale=alt.Scale(zero=False), axis=alt.Axis(title='Lap time')),
                    alt.Y('type3', scale=alt.Scale(zero=False), axis=alt.Axis(title='Session')),
                    color=alt.Color('fullname', legend=alt.Legend(title='The drivers')),
                    tooltip = [alt.Tooltip('fullname'), alt.Tooltip('q3')])
            .properties(height=700, width=700, title=f'The qualification of {grand_prix} {a}').interactive())

    @st.cache(allow_output_mutation=True)
    def pointsq():
        return (alt.Chart(df3).mark_point(size=150, filled=True, color='black')
            .encode(alt.X('time', scale=alt.Scale(zero=False), axis=alt.Axis(title='Lap time')),
                    alt.Y('session', scale=alt.Scale(zero=False), axis=alt.Axis(title='Session')),
                    tooltip = [alt.Tooltip('time'), alt.Tooltip('session')])
            .properties(height=700, width=700, title=f'The qualification of {grand_prix} {a}').interactive())


    st.altair_chart(figure1()+figure2()+figure3()+pointsq())

    st.write("### Гонка")

    """Гланым событием любого Гран-При является воскресная гонка. В этом проекте ей также будет посвящено больше всего 
    внимания."""

    """Во-первых, выберите, визуализацию какой гонки вы хотите посмотреть. Если вы хотите посмотреть, что происходило 
    в гонке на этапе, для которого Вы сейчас смотрели результаты квалификации, то выберете соответствующий вариант ниже.
    Если же вы хотите посмотреть на другую гонку, то можете выбрать другой вариант и, как и ранее, выбрать любой 
    доступный этап (кстати, гонки доступны не с 2006 года, а с 1996)."""

    question = st.radio('Вы хотите выбрать другой этап для визуализации гонки?', ['Нет, я хочу посмотреть визуализацию гонки того же этапа',
                                                                             'Да, я хочу выбрать другой этап'])

    if question == 'Да, я хочу выбрать другой этап':
        a = st.slider('Выберете сезон:', 1996, 2021)
        races_in_this_year = races[lambda x: x['year'] == a]
        if a:
            grand_prix = st.selectbox('Выберите Гран-При', races_in_this_year['name'].unique())

    """Теперь, когда гоночный этап выбран, перейдём к визуализациям. 
    На первом графике Вы можете увидеть динамику изменения позиций гонщиков в течение гонки (кстати, можно нажать на 
    фамилию гонщика справа и тогда будет выделене линия с его выступлением в гонке). Кроме того, что из этой визуализации 
    видно, кто когда и кого обгонял, можно увидеть ещё многое другое. Например, можно проанализировать, у кого получился 
    хороший старт, а у кого плохой (для этого нужно посомтреть, сколько позиций отыграл или проигралгонщик за первый круг).
    Также можно увидеть, если кто-то не доехал до финиша: тогда его линия прерывается где-то в течение гонки. 
    Если же линия немного не доходит до последнего круга, то скорее всего гонщик просто отстал на 1-2 круга и финишировал, 
    но с сильным отставанием от лидера."""

    """Я советую посмотреть эту визуализацию для разных гонок. А особенно рекомендую Гран-При Канады 2011, Гран-При Европы
    2012 и Гран-При Германии 2019. Эти гонки получились очень зрелищными и интересными для анализа: много обгонов, 
    неожиданные прорывы, сходы и аварии."""

    @st.cache(allow_output_mutation=True)
    def race_stat(a, grand_prix):
        prerace_df = lap_times.merge(races, left_on='raceId', right_on='raceId')[lambda x: x['year'] == a]
        race_df = prerace_df[lambda x: x['name'] == grand_prix].merge(drivers, left_on='driverId', right_on='driverId')[
            ['fullname', 'lap', 'position']]

        df_for_start1 = (results.merge(drivers, left_on='driverId', right_on='driverId')
            .merge(races, left_on='raceId', right_on='raceId')[lambda x: x['year'] == a])
        df_for_start2 = df_for_start1[lambda x: x['name'] == grand_prix][['fullname', 'grid']]

        for i in range(len(df_for_start2.index)):
            if df_for_start2.iloc[i]['grid'] > 0:
                race_df = race_df.append({'fullname': df_for_start2.iloc[i]['fullname'], 'lap': 0,
                                          'position': df_for_start2.iloc[i]['grid']}, ignore_index=True)
        return race_df

    race_df = race_stat(a, grand_prix)

    @st.cache(allow_output_mutation=True)
    def race_graph1():
        selection = alt.selection_multi(fields=['fullname'], bind='legend')

        return (alt.Chart(race_df).mark_line(point=True).encode(
            x=alt.X("lap", scale=alt.Scale(zero=False), title="Lap"),
            y=alt.Y("position", scale=alt.Scale(zero=False), sort='descending', axis=alt.Axis(title='Position')),
            color=alt.Color("fullname", legend=alt.Legend(title='The drivers')),
            tooltip=[alt.Tooltip('fullname'), alt.Tooltip('lap'), alt.Tooltip('position')],
            opacity=alt.condition(selection, alt.value(1), alt.value(0.2))
        ).properties(
            title=f"The race of {grand_prix} {a}",
            width=700,
            height=700,
        ).add_selection(selection).interactive())

    st.altair_chart(race_graph1())

    """Перейдём к следующей визуализации. Как Вы могли заметить ранее, в некоторых гонках лидер едет на первой позиции
    почти всю гонку, а иногда гонщик становится первым лишь на последних кругах. Поэтому интересной для анализа оказывается 
    следующая визуализация в виде тепловой карты. На ней показано, на каких позициях шёл каждый гонщик, причём чем ярче цвет,
    тем чаще гонщик оказывался на этой позиции. Предлагаю также посмотреть на такую визуализацию на разных примерах."""

    new_race_df1 = (race_df.pivot_table(values='lap', index='fullname', columns='position', aggfunc='count')
                   .fillna(0))
    new_race_df2 = (race_df.pivot_table(values='position', index='fullname', aggfunc='mean')
                   .fillna(0))
    new_race_df1['average'] = new_race_df2['position']

    new_race_df1 = new_race_df1.sort_values(by='average', ascending=True)

    fig, ax = plt.subplots()
    chart = sns.heatmap(data=new_race_df1.drop('average', axis=1),
                         ax=ax, cbar=True, cmap='Oranges', linewidths=.5)
    chart.set_title(f"The heatmap of the race of {grand_prix} {a}")
    chart.set(xlabel='Position', ylabel='Drivers')
    st.pyplot(fig)

    """На предыдущих визулизациях Вы могли посмотреть на ход гонки в целом и на то, как гонщики провели гонку друг 
    относительно друга. Далее я предалагаю выбрать одного пилота и посмотреть немного внимательнее на его выступление в гонке."""

    lap_times_gp = (lap_times.merge(races, left_on='raceId', right_on='raceId')[lambda x: x['year'] == a]
        .merge(drivers, left_on='driverId', right_on='driverId')[lambda x: x['name'] == grand_prix])

    the_driver = st.selectbox('Выберите гонщика:', lap_times_gp['fullname'].unique())

    driver_laps = lap_times_gp[lambda x: x['fullname'] == the_driver][
        ['fullname', 'lap', 'position', 'milliseconds']]

    """На следующей визуализации Вы можете посомтреть, с каким темпом шёл гонщик на протяжении гонки. 
    Точки на этом графике отражают то, за ккаое время гонщик прошёл тот или иной круг. Как Вы можете видеть, в большинстве
    случаев гонщик проходит гонку в одном темпе. Иногда темп к концу гонки растёт, так как запас топлива сокращается и 
    болид становится легче. Но также есть точки, лежащие сильно чёрной выше кривой, усредняющей темп гонщика. Эти точки 
    отражают ситуацию, когда с гонщиком или на трассе что-то произошло. Возможно, гонщик просто заехал на пит-стоп, чтобы
    заменить резину на колёсах. А, возможно, что на трассе произошла авария, из-за чего все гонщики должны снизить темп.
    Как и ранее, Вы можете посмотреть на то, как прошла гонка, для разных гонщиков и для разных этапов."""

    plot4 = (ggplot(driver_laps, aes(x='lap', y='milliseconds'))
             + geom_point(aes(color='milliseconds')) + geom_smooth() +
             labs(x="Lap", y="Time", title=f"The performance of {the_driver} in the race of {grand_prix} {a}",
                  color="Time (milliseconds)")
             )
    st.pyplot(ggplot.draw(plot4))

    st.write("## Часть 6: Анализ сезонов")

    """Каждый гонщик хочет выиграть в гонке Формулы-1, но главная цель - стать чемпионом мира по итогам не одной гонки, 
    а целого сезона. Поэтому следующим (и последним) объектом моего небольшого исследования станут результаты различных 
    сезонов Формулы-1."""

    """Как всегда, начнём с выбора сезона."""

    b = st.slider('Выберите год:', 1950, 2021)

    """Теперь приступим к визуализации."""

    """Многие гонщики рассматривают гоночный сезон как одну большую гонку. Я предлагаю поступить так же и часть 
    визуализаций сделать похожими на те, что использовались для анализа гонки. А именно, я предлагаю рассмотреть, как
    менялись позиции в чемпионате на протяжении сезона. Но помимо того, как гонщики менялись позициями, я предлагаю 
    посмотреть и на то, как у них менялись набранные очки."""

    """Как и раньше, я советую посмотреть эти визуализации для разных сезонов. Если Вы это сделаете, то Вы можете
    заметить одну тенденцию: количество участников Гран-При уменьшается при приближении к настоящему времени. 
    Это объясняется тем, что сначала Формула-1 была очень непопулрным спортом и гонщиками мог стать любой достаточно 
    богатый человек, который не боялся попасть в аварию на огромной скорости. А сейчас чемпионат Формулы-1 - это 20
    лучших гонщиков планеты, которые проедлали огромный путь, чтобы попасть в Формулу-1."""

    """Кстати, советую посмотреть визуализацию сезонов 2007, 2010, 2012, 2021. В эти годы борьба за первое место была до 
    последнего Гран-При, а иногда даже до последнего круга."""

    table_of_standings = driver_standings.merge(races, left_on='raceId', right_on='raceId')[lambda x: x['year']==b]\
        .merge(drivers, left_on='driverId', right_on='driverId')[['fullname', 'round', 'position', 'points']]

    selection1 = alt.selection_multi(fields=['fullname'], bind='legend')

    graph1 = alt.Chart(table_of_standings).mark_line(point=True).encode(
        x=alt.X("round", scale=alt.Scale(zero=False), title="Round"),
        y=alt.Y("position", scale=alt.Scale(zero=False), sort='descending', axis=alt.Axis(title='Position')),
        color=alt.Color("fullname", legend=alt.Legend(title='The drivers')), tooltip=[alt.Tooltip('fullname'), alt.Tooltip('position'), alt.Tooltip('points')],
        opacity=alt.condition(selection1, alt.value(1), alt.value(0.2))
    ).properties(
        title=f"The championship battle in {b} (positions)",
        width=550,
        height=550,
    ).add_selection(selection1).interactive()

    for i in range(len(table_of_standings['fullname'].unique())):
        table_of_standings = table_of_standings.append({'fullname': table_of_standings['fullname'].unique()[i], 'round': 0,
                                      'position': 0, 'points': 0}, ignore_index=True)

    graph2 = alt.Chart(table_of_standings).mark_line(point=True).encode(
        x=alt.X("round", scale=alt.Scale(zero=False), title="Round"),
        y=alt.Y("points", scale=alt.Scale(zero=False), sort='ascending', axis=alt.Axis(title='Points')),
        color=alt.Color("fullname", legend=alt.Legend(title='The drivers')), tooltip=[alt.Tooltip('fullname'), alt.Tooltip('position'), alt.Tooltip('points')],
        opacity=alt.condition(selection1, alt.value(1), alt.value(0.2))
    ).properties(
        title=f"The championship battle in {b} (points)",
        width=550,
        height=550,
    ).add_selection(selection1).interactive()

    st.altair_chart(graph1 & graph2)

    """В заключение, предлагю посмотреть и на кубок конструкторов. В этот раз для разнообразия обойдёмся без динамики, а 
    посмотрим лишь на сами итоги и увидим, какая команда стала чемпионом в выбранный год."""

    year = st.slider('Выберите год:', 1958, 2021)
    fig, ax = plt.subplots()
    data = constructor_standings[lambda x: x['Year']==year]
    chart = sns.barplot(data=data, x='PTS', y='Team',
                         ax=ax, palette='flare')
    chart.bar_label(chart.containers[0], fontsize=8.5, color='black')
    chart.set_title(f"World Constructors' Championship in {year}")
    chart.set(xlabel='Points', ylabel='Teams')
    st.pyplot(fig)

    """Спасибо за внимание! Надеюсь, мой проект оказался интересным!"""












