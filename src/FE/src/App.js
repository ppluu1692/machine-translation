import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function Translation() {
  const [inputText, setInputText] = useState('');
  const [translatedText, setTranslatedText] = useState('');

  const [language, setLanguage] = useState(true)

  const translateText = async () => {
    try {
      
      if (language) {
        //API eng ->vie
        let myObject = await fetch(`http://localhost:5000/en-to-vi?message="${inputText}"`);
        let myText = await myObject.text();
        setTranslatedText(myText);
      }
      else {
        //API vie -> eng
        let myObject = await fetch(`http://localhost:5000/vi-to-en?message="${inputText}"`);
        let myText = await myObject.text();
        setTranslatedText(myText);
      }
    } catch (error) {
      console.log(error);
    }
  };

  return (
    <>
      <div className="container">
        <div className="wrapper">
          <div className="text-input">
            <textarea
              spellCheck="false"
              className="from-text"
              placeholder="Enter text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
            ></textarea>
            <textarea
              spellCheck="false"
              readOnly
              disabled
              className="to-text"
              placeholder="Translation"
              value={translatedText}
            ></textarea>
          </div>
        </div>
        <div className="text-note">
          <div className='text-english'>
            {language ? "English" : "Vietnamese"}
          </div>
          <div className="change">
            <button className="button-change" onClick={() => { setLanguage(!language) }}>
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-arrow-left-right" viewBox="0 0 16 16">
                <path fill-rule="evenodd" d="M1 11.5a.5.5 0 0 0 .5.5h11.793l-3.147 3.146a.5.5 0 0 0 .708.708l4-4a.5.5 0 0 0 0-.708l-4-4a.5.5 0 0 0-.708.708L13.293 11H1.5a.5.5 0 0 0-.5.5zm14-7a.5.5 0 0 1-.5.5H2.707l3.147 3.146a.5.5 0 1 1-.708.708l-4-4a.5.5 0 0 1 0-.708l4-4a.5.5 0 1 1 .708.708L2.707 4H14.5a.5.5 0 0 1 .5.5z" />
              </svg>
            </button>
          </div>
          <div className='text-vietnamese'>
            {language ? "Vietnamese" : "English"}
          </div>
        </div>
        <button className="button-translate" onClick={translateText}>TRANSLATE</button>
      </div>
    </>
  );
}

export default Translation;