﻿using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.IO;
using System.Text;

namespace MachineLearning.Data
{
   /// <summary>
   /// Classe per lo storage di dati di tipo testo in memoria
   /// </summary>
   [Serializable]
   public sealed class DataStorageTextMemory : DataStorageText, IDataTextProvider, IDataTimestamp
   {
      #region Fields
      /// <summary>
      /// Dati testuali
      /// </summary>
      private string _textData;
      #endregion
      #region Properties
      /// <summary>
      /// Dati testuali
      /// </summary>
      public string TextData { get => _textData; set { _textData = value; DataTimestamp = DateTime.UtcNow; } }
      /// <summary>
      /// Data e ora dell'oggetto
      /// </summary>
      public DateTime DataTimestamp { get; private set; }
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="options">Opzioni di caricamento dei testi</param>
      public DataStorageTextMemory(TextLoader.Options options) : base(options) { }
      /// <summary>
      /// Restituisce uno stream leggibile.
      /// </summary>
      /// <returns>Lo stream di lettura</returns>
      protected override Stream GetReadStream() => new MemoryStream(Encoding.Default.GetBytes(TextData ?? ""));
      /// <summary>
      /// Salva i dati
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <param name="data">L'accesso ai dati</param>
      public override void SaveData(IMachineLearningContextProvider context, IDataView data)
      {
         MachineLearningContext.CheckMLNET(context, nameof(context));
         // Data e ora
         var timestamp = DateTime.UtcNow;
         lock (this) {
            // Oggetto per la scrittura dei dati in memoria
            using var stream = new MemoryStream();
            SaveTextData(context, data, stream, false);
            // Crea uno stream per la lettura
            stream.Position = 0;
            using var reader = new StreamReader(stream);
            // Aggiorna la stringa
            TextData = reader.ReadToEnd();
            reader.Close();
            stream.Close();
            DataTimestamp = timestamp;
         }
      }
      #endregion
   }
}
