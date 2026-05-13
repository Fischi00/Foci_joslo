import React, { useState, useEffect } from 'react';

function App() {
  const [teamsByLeague, setTeamsByLeague] = useState({});
  
  // Ligák és Csapatok állapotai külön-külön
  const [homeLeague, setHomeLeague] = useState('');
  const [awayLeague, setAwayLeague] = useState('');
  const [home, setHome] = useState('');
  const [away, setAway] = useState('');
  
  const [model1, setModel1] = useState('XGBoost');
  const [model2, setModel2] = useState('NeuralNet');
  const [result1, setResult1] = useState(null);
  const [result2, setResult2] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetch('http://127.0.0.1:8000/teams')
      .then(res => res.json())
      .then(data => {
        setTeamsByLeague(data);
        const leagues = Object.keys(data);
        if (leagues.length > 0) {
          // Kezdőértékek beállítása az első elérhető ligára és csapatokra
          setHomeLeague(leagues[0]);
          setAwayLeague(leagues[0]);
          setHome(data[leagues[0]][0]);
          setAway(data[leagues[0]][1] || data[leagues[0]][0]);
        }
      }).catch(err => console.error("Hiba a csapatok betöltésekor", err));
  }, []);

  // Ha változik a hazai liga, frissítjük a hazai csapatot az új liga első csapatára
  const handleHomeLeagueChange = (e) => {
    const newLeague = e.target.value;
    setHomeLeague(newLeague);
    setHome(teamsByLeague[newLeague][0]);
  };

  // Ha változik a vendég liga, frissítjük a vendég csapatot
  const handleAwayLeagueChange = (e) => {
    const newLeague = e.target.value;
    setAwayLeague(newLeague);
    setAway(teamsByLeague[newLeague][0]);
  };

  const handlePredict = async () => {
    setLoading(true);
    setResult1(null); setResult2(null);
    try {
      const res1 = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ home, away, model: model1 }),
      });
      const data1 = await res1.json();
      if (data1.status === 'success') setResult1(data1.prediction);

      const res2 = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ home, away, model: model2 }),
      });
      const data2 = await res2.json();
      if (data2.status === 'success') setResult2(data2.prediction);
    } catch (error) {
      setResult1("Szerver hiba!"); setResult2("Szerver hiba!");
    }
    setLoading(false);
  };

  const leagueOptions = Object.keys(teamsByLeague).map(l => <option key={l} value={l}>{l}</option>);

  return (
    <div style={{ 
      minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', 
      backgroundImage: 'url("https://images.unsplash.com/photo-1518605368461-1ee12523f20e?q=80&w=2000&auto=format&fit=crop")', 
      backgroundSize: 'cover', backgroundAttachment: 'fixed', fontFamily: 'sans-serif' 
    }}>
      <div style={{ 
        width: '100%', maxWidth: '700px', padding: '30px', backgroundColor: 'rgba(255, 255, 255, 0.96)', 
        boxShadow: '0 10px 40px rgba(0,0,0,0.4)', borderRadius: '20px' 
      }}>
        <h1 style={{ textAlign: 'center', color: '#2c3e50', marginBottom: '25px' }}>⚽ Modell Párbaj: Liga Választóval</h1>

        <div style={{ display: 'flex', gap: '20px', marginBottom: '25px' }}>
          {/* Hazai szekció */}
          <div style={{ flex: 1, padding: '15px', border: '1px solid #ddd', borderRadius: '10px', backgroundColor: '#f9f9f9' }}>
            <h4 style={{ marginTop: 0, color: '#27ae60' }}>🏠 Hazai Oldal</h4>
            <label style={{ fontSize: '12px', fontWeight: 'bold' }}>Bajnokság:</label>
            <select value={homeLeague} onChange={handleHomeLeagueChange} style={{ width: '100%', padding: '8px', marginBottom: '10px', borderRadius: '5px' }}>
              {leagueOptions}
            </select>
            <label style={{ fontSize: '12px', fontWeight: 'bold' }}>Csapat:</label>
            <select value={home} onChange={(e) => setHome(e.target.value)} style={{ width: '100%', padding: '8px', borderRadius: '5px' }}>
              {homeLeague && teamsByLeague[homeLeague].map(t => <option key={t} value={t}>{t}</option>)}
            </select>
          </div>

          {/* Vendég szekció */}
          <div style={{ flex: 1, padding: '15px', border: '1px solid #ddd', borderRadius: '10px', backgroundColor: '#f9f9f9' }}>
            <h4 style={{ marginTop: 0, color: '#e67e22' }}>🚀 Vendég Oldal</h4>
            <label style={{ fontSize: '12px', fontWeight: 'bold' }}>Bajnokság:</label>
            <select value={awayLeague} onChange={handleAwayLeagueChange} style={{ width: '100%', padding: '8px', marginBottom: '10px', borderRadius: '5px' }}>
              {leagueOptions}
            </select>
            <label style={{ fontSize: '12px', fontWeight: 'bold' }}>Csapat:</label>
            <select value={away} onChange={(e) => setAway(e.target.value)} style={{ width: '100%', padding: '8px', borderRadius: '5px' }}>
              {awayLeague && teamsByLeague[awayLeague].map(t => <option key={t} value={t}>{t}</option>)}
            </select>
          </div>
        </div>

        {/* Modell választók */}
        <div style={{ display: 'flex', gap: '15px', marginBottom: '25px', padding: '15px', backgroundColor: '#34495e', borderRadius: '10px', color: 'white' }}>
          <div style={{ flex: 1 }}>
            <label style={{ fontSize: '12px' }}>Választott Modell A:</label>
            <select value={model1} onChange={(e) => setModel1(e.target.value)} style={{ width: '100%', padding: '5px', borderRadius: '5px' }}>
              <option value="XGBoost">XGBoost</option>
              <option value="RandomForest">Random Forest</option>
              <option value="Logistic">Logistic Reg.</option>
              <option value="NeuralNet">Neural Network</option>
            </select>
          </div>
          <div style={{ flex: 1 }}>
            <label style={{ fontSize: '12px' }}>Választott Modell B:</label>
            <select value={model2} onChange={(e) => setModel2(e.target.value)} style={{ width: '100%', padding: '5px', borderRadius: '5px' }}>
              <option value="XGBoost">XGBoost</option>
              <option value="RandomForest">Random Forest</option>
              <option value="Logistic">Logistic Reg.</option>
              <option value="NeuralNet">Neural Network</option>
            </select>
          </div>
        </div>

        <button onClick={handlePredict} disabled={loading} style={{ 
          width: '100%', padding: '15px', backgroundColor: '#2ecc71', color: 'white', border: 'none', 
          borderRadius: '10px', fontSize: '18px', fontWeight: 'bold', cursor: 'pointer', boxShadow: '0 4px 0 #27ae60' 
        }}>
          {loading ? '🔮 Elemzés folyamatban...' : 'Jóslat Futtatása 🚀'}
        </button>

        {result1 && result2 && (
          <div style={{ display: 'flex', gap: '15px', marginTop: '25px' }}>
            <div style={{ flex: 1, padding: '15px', backgroundColor: '#fff', borderRadius: '10px', textAlign: 'center', borderTop: '5px solid #e74c3c', boxShadow: '0 4px 10px rgba(0,0,0,0.1)' }}>
              <small>{model1}</small>
              <h3 style={{ margin: '5px 0 0 0' }}>{result1}</h3>
            </div>
            <div style={{ flex: 1, padding: '15px', backgroundColor: '#fff', borderRadius: '10px', textAlign: 'center', borderTop: '5px solid #3498db', boxShadow: '0 4px 10px rgba(0,0,0,0.1)' }}>
              <small>{model2}</small>
              <h3 style={{ margin: '5px 0 0 0' }}>{result2}</h3>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;