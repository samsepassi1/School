# flake8: noqa: F403, F405
from fasthtml.common import *
import matplotlib.pyplot as plt

from employee_events import Employee, Team

# import the load_model function from the utils.py file
from utils import load_model

"""
Below, we import the parent classes
you will use for subclassing
"""
from base_components import (
    Dropdown,
    BaseComponent,
    Radio,
    MatplotlibViz,
    DataTable
)

from combined_components import FormGroup, CombinedComponent


# Create a subclass of base_components/dropdown
# called `ReportDropdown`
class ReportDropdown(Dropdown):

    # Overwrite the build_component method
    def build_component(self, entity_id, model):
        # Set the `label` attribute to the model's `name` attribute
        self.label = model.name

        # Return the output from the parent class's build_component method
        return super().build_component(entity_id, model)

    # Overwrite the `component_data` method
    def component_data(self, entity_id, model):
        # call the employee_events method that returns names and ids
        return model.names()


# Create a subclass of base_components/BaseComponent
# called `Header`
class Header(BaseComponent):

    # Overwrite the `build_component` method
    def build_component(self, entity_id, model):

        # return a fasthtml H1 object containing the model's name attribute
        return H1(model.name)


# Create a subclass of base_components/MatplotlibViz
# called `LineChart`
class LineChart(MatplotlibViz):

    # Overwrite the parent class's `visualization` method
    def visualization(self, asset_id, model):

        # Pass the `asset_id` argument to the model's `event_counts` method
        df = model.event_counts(asset_id)

        # Use the pandas .fillna method to fill nulls with 0
        df = df.fillna(0)

        # Use the pandas .set_index method to set the date column as the index
        df = df.set_index('event_date')

        # Sort the index
        df = df.sort_index()

        # Use the .cumsum method to change the data to cumulative counts
        df = df.cumsum()

        # Set the dataframe columns to the list ['Positive', 'Negative']
        df.columns = ['Positive', 'Negative']

        # Initialize a pandas subplot
        fig, ax = plt.subplots()

        # call the .plot method for the cumulative counts dataframe
        df.plot(ax=ax)

        # pass the axis variable to the `.set_axis_styling` method
        self.set_axis_styling(ax, bordercolor='black', fontcolor='black')

        # Set title and labels for x and y axis
        ax.set_title('Cumulative Events', fontsize=20)
        ax.set_xlabel('Date')
        ax.set_ylabel('Events')


# Create a subclass of base_components/MatplotlibViz
# called `BarChart`
class BarChart(MatplotlibViz):

    # Create a `predictor` class attribute
    predictor = load_model()

    # Overwrite the parent class `visualization` method
    def visualization(self, asset_id, model):

        # pass the `asset_id` to the `.model_data` method
        data = model.model_data(asset_id)

        # pass the data to the `predict_proba` method
        preds = self.predictor.predict_proba(data)

        # Index the second column of predict_proba output
        preds = preds[:, 1]

        # If the model's name attribute is "team"
        # visualize the mean of the predict_proba output
        if model.name == "team":
            pred = preds.mean()

        # Otherwise set `pred` to the first value
        else:
            pred = preds[0]

        # Initialize a matplotlib subplot
        fig, ax = plt.subplots()

        # Run the following code unchanged
        ax.barh([''], [pred])
        ax.set_xlim(0, 1)
        ax.set_title('Predicted Recruitment Risk', fontsize=20)

        # pass the axis variable to the `.set_axis_styling` method
        self.set_axis_styling(ax, bordercolor='black', fontcolor='black')

# Create a subclass of combined_components/CombinedComponent
# called Visualizations


class Visualizations(CombinedComponent):

    # Set the `children` class attribute
    children = [LineChart(), BarChart()]

    # Leave this line unchanged
    outer_div_type = Div(cls='grid')

# Create a subclass of base_components/DataTable
# called `NotesTable`


class NotesTable(DataTable):

    # Overwrite the `component_data` method
    def component_data(self, entity_id, model):

        # pass the entity_id to the model's .notes method. Return the output
        return model.notes(entity_id)


class DashboardFilters(FormGroup):

    id = "top-filters"
    action = "/update_data"
    method = "POST"

    children = [
        Radio(
            values=["Employee", "Team"],
            name='profile_type',
            hx_get='/update_dropdown',
            hx_target='#selector'
        ),
        ReportDropdown(
            id="selector",
            name="user-selection")
    ]

# Create a subclass of CombinedComponents called `Report`


class Report(CombinedComponent):

    # Set the `children` class attribute
    children = [Header(), DashboardFilters(), Visualizations(), NotesTable()]


# Initialize a fasthtml app
app, rt = fast_app()

# Initialize the `Report` class
report = Report()


# Create a route for a get request — root path
@app.get('/')
def index():

    # Call the initialized report
    # pass the integer 1 and an instance of the Employee class
    return report(1, Employee())

# Create a route for a get request — employee by ID


@app.get('/employee/{id:str}')
def employee(id: str):

    # Call the initialized report with ID and Employee instance
    return report(id, Employee())

# Create a route for a get request — team by ID


@app.get('/team/{id:str}')
def team(id: str):

    # Call the initialized report with id and Team instance
    return report(id, Team())


# Keep the below code unchanged!
@app.get('/update_dropdown{r}')
def update_dropdown(r):
    dropdown = DashboardFilters.children[1]
    print('PARAM', r.query_params['profile_type'])
    if r.query_params['profile_type'] == 'Team':
        return dropdown(None, Team())
    elif r.query_params['profile_type'] == 'Employee':
        return dropdown(None, Employee())


@app.post('/update_data')
async def update_data(r):
    from fasthtml.common import RedirectResponse
    data = await r.form()
    profile_type = data._dict['profile_type']
    id = data._dict['user-selection']
    if profile_type == 'Employee':
        return RedirectResponse(f"/employee/{id}", status_code=303)
    elif profile_type == 'Team':
        return RedirectResponse(f"/team/{id}", status_code=303)


serve()
