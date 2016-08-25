import pytest
from bambi.models import Term, Model
from os.path import dirname, join
import pandas as pd


@pytest.fixture(scope="module")
def diabetes_data():
    from os.path import dirname, join
    data_dir = join(dirname(__file__), 'data')
    data = pd.read_csv(join(data_dir, 'diabetes.txt'), sep='\t')
    data['age_grp'] = 0
    data['age_grp'][data['AGE'] > 40] = 1
    data['age_grp'][data['AGE'] > 60] = 2
    return data


@pytest.fixture(scope="module")
def base_model(diabetes_data):
    return Model(diabetes_data)


def test_term_init(diabetes_data):
    model = Model(diabetes_data)
    term = Term(model, 'BMI', diabetes_data['BMI'])
    # Test that all defaults are properly initialized
    assert term.name == 'BMI'
    assert term.categorical == False
    assert term.type_ == 'fixed'
    assert term.levels is not None
    assert term.data.shape == (442, 1)


def test_term_split(diabetes_data):
    # Split a continuous fixed variable
    model = Model(diabetes_data)
    model.add_term('BMI', split_by='age_grp')
    assert model.terms['BMI'].data.shape == (442, 3)
    # Split a categorical fixed variable
    model.reset()
    model.add_term('BMI', split_by='age_grp', categorical=True)
    assert model.terms['BMI'].data.shape == (442, 489)
    # Split a continuous random variable
    model.reset()
    model.add_term('BMI', split_by='age_grp', categorical=False, random=True)
    assert model.terms['BMI'].data.shape == (442, 3)
    # Split a categorical random variable
    model.reset()
    model.add_term('BMI', split_by='age_grp', categorical=True, random=True)
    t = model.terms['BMI'].data
    assert isinstance(t, dict)
    assert t['age_grp[0]'].shape == (442, 83)


def test_model_init_and_intercept(diabetes_data):

    model = Model(diabetes_data, intercept=True)
    assert hasattr(model, 'data')
    assert 'Intercept' in model.terms
    assert len(model.terms) == 1
    assert model.y is None
    assert hasattr(model, 'backend')
    model = Model(diabetes_data)
    assert 'Intercept' not in model.terms
    assert not model.terms


def test_add_term_to_model(base_model):

    base_model.add_term('BMI')
    assert isinstance(base_model.terms['BMI'], Term)
    base_model.add_term('age_grp', random=False, categorical=True)
    # Test that arguments are passed appropriately onto Term initializer
    base_model.add_term('BP', random=True, split_by='age_grp', categorical=True)
    assert isinstance(base_model.terms['BP'], Term)


def test_one_shot_formula_fit(base_model):
    base_model.fit('BMI ~ S1 + S2', samples=50)
    nv = base_model.backend.model.named_vars
    targets = ['likelihood', 'b_S1', 'likelihood_sd_log_', 'b_Intercept']
    assert len(set(nv.keys()) & set(targets)) == 4
    assert len(base_model.backend.trace) == 50


def test_invalid_chars_in_random_effect(base_model):
    with pytest.raises(ValueError):
        base_model.fit(random=['1+BP|age_grp'])