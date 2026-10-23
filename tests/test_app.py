from plotly_chess.app import create_app

def test_app_layout_and_callback_registration():
    app = create_app()
    client = app.server.test_client()
    assert client.get('/').status_code == 200
    assert client.get('/_dash-layout').status_code == 200
    assert client.get('/_dash-dependencies').status_code == 200
    assert len(app.callback_map) == 1

def test_callback_move_and_reset():
    app = create_app()
    client = app.server.test_client()
    output = next(iter(app.callback_map))
    inputs = [
        {'id':'board-action','property':'data','value':None},
        {'id':'play','property':'n_clicks','value':1},
        {'id':'reset','property':'n_clicks','value':0},
        {'id':'hint','property':'n_clicks','value':0},
        {'id':'mode','property':'value','value':'local'}]
    def request(state, trigger):
        response = client.post('/_dash-update-component',json={
            'output':output,
            'outputs':[{'id':o.component_id,'property':o.component_property} for o in app.callback_map[output]['output']],
            'inputs':inputs,
            'state':[{'id':'notation','property':'value','value':'e4'},
                     {'id':'promotion','property':'value','value':'q'},
                     {'id':'game','property':'data','value':state}],
            'changedPropIds':[trigger]})
        assert response.status_code == 200
        return response.get_json()['response']
    moved = request({'moves':[],'selected':None},'play.n_clicks')
    assert moved['game']['data']['moves'] == ['e2e4']
    reset = request(moved['game']['data'],'reset.n_clicks')
    assert reset['game']['data']['moves'] == []

    # Drag-and-drop is validated on the server independently of client hints.
    inputs[0]['value'] = {'type':'drop','from':12,'to':28}
    dropped = request({'moves':[],'selected':None},'board-action.data')
    assert dropped['game']['data']['moves'] == ['e2e4']
    inputs[0]['value'] = {'type':'drop','from':12,'to':36}
    invalid = request({'moves':[],'selected':None},'board-action.data')
    assert invalid['game']['data']['moves'] == []
    assert 'not available' in invalid['feedback']['children']
    inputs[0]['value'] = {'type':'click','square':12}
    selected = request({'moves':[],'selected':None},'board-action.data')
    assert selected['game']['data']['selected'] == 12
    inputs[0]['value'] = {'type':'click','square':28}
    clicked = request(selected['game']['data'],'board-action.data')
    assert clicked['game']['data']['moves'] == ['e2e4']
