# Dictionary for Undergraduate Regulations
undergrad_regulation_pages = [
    {
        'id': 'ug-sep2020-null',  # Regulations from September 2020
        'start_year': 2020,
        'end_year': None
    },
    {
        'id': 'ug-sep2019-sep2020',  # Regulations from September 2019 to September 2020
        'start_year': 2019,
        'end_year': 2020
    },
    {
        'id': 'ug-sep2017-sep2019',  # Regulations from September 2017 to September 2019
        'start_year': 2017,
        'end_year': 2019
    },
    {
        'id': 'ug-sep2015-sep2017',  # Regulations from September 2015 to September 2017
        'start_year': 2015,
        'end_year': 2017
    },
    {
        'id': 'ug-sep2005-sep2015',  # Regulations from September 2005 to September 2015
        'start_year': 2005,
        'end_year': 2015
    },
        {
        'id': 'ug-null-2005',  # Regulations for first degrees before September 2005
        'start_year': None,
        'end_year': 2005
    },
    {
        'id': 'ug-nonHonours-2015-null',  # Regulations for students who do not meet Honours requirements from September 2015
        'start_year': 2015,
        'end_year': None,
        'type': 'honours'
    },
    {
        'id': 'ug-nonHonours-null-2015',  # Regulations for students who did not meet Honours requirements before September 2015
        'start_year': None,
        'end_year': 2015,
        'type': 'honours'
    }
]

# Dictionary for Postgraduate Regulations
postgrad_regulation_pages = [
    {
        'id': 'pg-sep2021-null',  # Regulations for students admitted from September 2021
        'start_year': 2021,
        'end_year': None
    },
    {
        'id': 'pg-sep2020-2021',  # Regulations for students admitted from September 2020
        'start_year': 2020,
        'end_year': 2021
    },
    {
        'id': 'pg-sep2019-sep2020',  # Regulations for students admitted from September 2019 and before September 2020
        'start_year': 2019,
        'end_year': 2020
    },
    {
        'id': 'pg-sep2017-sep2019',  # Regulations for students admitted from September 2017 and before September 2019
        'start_year': 2017,
        'end_year': 2019
    },
    {
        'id': 'pg-sep2016-sep2017',  # Regulations for students admitted from September 2016 and before September 2017
        'start_year': 2016,
        'end_year': 2017
    },
    {
        'id': 'pg-sep2012-sep2016',  # Regulations for students admitted from September 2012 and before September 2016
        'start_year': 2012,
        'end_year': 2016
    },
    {
        'id': 'pg-sep2006-sep2012',  # Regulations for students admitted from September 2006 and before September 2012
        'start_year': 2006,
        'end_year': 2012
    },
    {
        'id': 'pg-null-sep2006',  # Regulations before September 2006
        'start_year': None,
        'end_year': 2006
    }
]

def get_regulation_page(year, program='undergraduate', student_type='regular'):
    """
    Returns the regulation page id based on the provided year, program type,
    and optional student_type (e.g., 'honours' for undergraduates).
    """
    pages = None

    if program == 'undergraduate':
        pages = undergrad_regulation_pages
        # Handle honours cases separately for undergraduate students.
        if student_type == 'non-honours':
            if year >= 2015:
                return next(page['id'] for page in pages if page.get('id') == 'ug-nonHonours-2015-null')
            else:
                return next(page['id'] for page in pages if page.get('id') == 'ug-nonHonours-null-2015')
    elif program == 'postgraduate':
        pages = postgrad_regulation_pages
    else:
        return 'unknown program'

    # Iterate through the pages to find the one that fits the input year.
    for page in pages:
        start_year = page.get('start_year')
        end_year = page.get('end_year')

        if start_year is not None and end_year is None:
            # Covers years from start_year onward.
            if year >= start_year:
                return page['id']
        elif start_year is not None and end_year is not None:
            # Covers the range [start_year, end_year)
            if start_year <= year < end_year:
                return page['id']
        elif start_year is None and end_year is not None:
            # Covers any year less than end_year.
            if year < end_year:
                return page['id']

    return 'regulation-not-found'