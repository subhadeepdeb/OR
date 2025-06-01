Sets
i 'routes' /1*5/
j 'months' /1*6/
k 'aircraft_types' /1*3/
e 'employee_types' /pilot, copilot, attendant/;

Parameters
demand(i,j) 'Passenger demand for each route and month'
capacity(k) 'Aircraft capacity'
owned_fleet(k) 'Number of owned aircraft of each type'
leasing_cost(k) 'Monthly leasing cost for each aircraft type'
crew_req(k,e) 'Crew requirements for each aircraft type'
flight_duration(i) 'Flight duration for each route'
fuel_cost(i) 'Fuel cost for each route'
salary(e) 'Monthly salary for each employee type'
hiring_cost(e) 'Hiring cost for each employee type'
firing_cost(e) 'Firing cost for each employee type'
training_cost(e) 'Training cost for each employee type'
initial_staff(e) 'Initial number of staff for each employee type'
max_hours 'Working hours limit per employee per month' /160/;

$include data.gms

Variables
Z 'Total cost';

Integer Variables
X(i,j,k) 'Number of flights on route i in month j using aircraft type k'
leased_fleet(k,j) 'Number of leased aircraft of type k in month j'
staff(e,j) 'Number of staff of type e in month j'
hire(e,j) 'Number of staff of type e hired in month j'
fire(e,j) 'Number of staff of type e fired in month j';

Equations
objective  Objective Function
meet_demand(i,j) Passenger Demand
use_all_type1(j) Type-1 aircraft Utilization
additional_demand(i,j) Demand Satisfaction
aircraft_availability_other(k,j) Aircraft Balance
crew_requirements(e,j) Crew Satisfaction
working_hours_limit(e,j) Maximum Work Hours for each employee
staff_balance(e,j) Workforce Balance
initial_staff_constraint(e) Initial staff levels;

objective.. Z =E= sum((i,j,k), X(i,j,k) * fuel_cost(i))
                + sum((k,j)$(ord(k) > 1), leased_fleet(k,j) * leasing_cost(k))
                + sum((e,j), staff(e,j) * salary(e))
                + sum((e,j), hire(e,j) * (hiring_cost(e) + training_cost(e)))
                + sum((e,j), fire(e,j) * firing_cost(e));

meet_demand(i,j).. sum(k, X(i,j,k) * capacity(k)) =G= demand(i,j);

use_all_type1(j).. sum(i, X(i,j,'1')) =E= owned_fleet('1');

additional_demand(i,j).. sum(k$(ord(k) > 1), X(i,j,k) * capacity(k)) =G= max(0, demand(i,j) - owned_fleet('1') * capacity('1'));

aircraft_availability_other(k,j)$(ord(k) > 1).. sum(i, X(i,j,k)) =L= leased_fleet(k,j);

crew_requirements(e,j).. sum((i,k), X(i,j,k) * crew_req(k,e)) =L= staff(e,j);

working_hours_limit(e,j).. sum((i,k), X(i,j,k) * flight_duration(i) * crew_req(k,e)) =L= staff(e,j) * max_hours;

staff_balance(e,j)$(ord(j) > 1).. staff(e,j) =E= staff(e,j-1) + hire(e,j) - fire(e,j);

initial_staff_constraint(e).. staff(e,'1') =E= initial_staff(e) + hire(e,'1') - fire(e,'1');

Model demand_driven /all/;

Solve demand_driven using mip minimizing Z;

Display X.l, leased_fleet.l, staff.l, hire.l, fire.l, Z.l;
