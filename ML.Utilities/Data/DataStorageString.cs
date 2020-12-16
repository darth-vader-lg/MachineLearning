﻿using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.IO;
using System.Linq;

namespace ML.Utilities.Data
{
   /// <summary>
   /// Classe per lo storage di dati di tipo stringhe
   /// </summary>
   [Serializable]
   public sealed partial class DataStorageString : IDataStorage, IDataTextProvider, ITextOptionsProvider
   {
      #region Properties
      /// <summary>
      /// Configurazione dei dati
      /// </summary>
      public TextLoader.Options TextOptions { get; set; } = new TextLoader.Options();
      /// <summary>
      /// Dati testuali
      /// </summary>
      public string TextData { get; set; }
      #endregion
      #region Methods
      /// <summary>
      /// Carica i dati
      /// </summary>
      /// <param name="mlContext">Contesto di machine learning</param>
      /// <returns>L'accesso ai dati</returns>
      public IDataView LoadData(MLContext mlContext) => mlContext.Data.CreateTextLoader(TextOptions ?? new TextLoader.Options()).Load(new Source(TextData));
      /// <summary>
      /// Salva i dati
      /// </summary>
      /// <param name="mlContext">Contesto di machine learning</param>
      /// <param name="data">L'accesso ai dati</param>
      public void SaveData(MLContext mlContext, IDataView data)
      {
         // Oggetto per la scrittura dei dati in memoria
         using var writer = new MemoryStream();
         // Opzioni
         var opt = TextOptions ?? new TextLoader.Options();
         // Separatore di colonne
         var separator = opt.Separators?.FirstOrDefault() ?? '\t';
         separator = separator != default ? separator : '\t';
         // Salva come testo i dati
         mlContext.Data.SaveAsText(data, writer, separator, opt.HasHeader, false, false, false);
         // Crea uno stream per la lettura
         writer.Position = 0;
         using var reader = new StreamReader(writer);
         // Aggiorna la stringa
         TextData = reader.ReadToEnd();
      }
      #endregion
   }

   /// <summary>
   /// La sorgente dei dati
   /// </summary>
   public partial class DataStorageString
   {
      private class Source : IMultiStreamSource
      {
         #region Fields
         /// <summary>
         /// Testo
         /// </summary>
         private readonly string text;
         #endregion
         #region Properties
         /// <summary>
         /// Il numero di items
         /// </summary>
         public int Count => 1;
         #endregion
         #region Methods
         /// <summary>
         /// Costruttore
         /// </summary>
         /// <param name="text">Testo</param>
         public Source(string text) => this.text = text;
         /// <summary>
         /// Restituisce una stringa rappresentante il "path" dello stream indicato da index. Potrebbe essere null.
         /// </summary>
         /// <param name="index">L'indice dell'item</param>
         /// <returns>Sempre null</returns>
         public string GetPathOrNull(int index) => default;
         /// <summary>
         /// Apre l'item indicato e ne restituisce uno stream leggibile.
         /// </summary>
         /// <param name="index">L'indice dell'item</param>
         /// <returns>Lo stream di lettura</returns>
         public Stream Open(int index)
         {
            var memoryStream = new MemoryStream();
            using var writer = new StreamWriter(memoryStream);
            writer.Write(text ?? "");
            memoryStream.Position = 0;
            return memoryStream;
         }
         /// <summary>
         /// Apre l'item indicato e ne restituisce uno stream di stringhe leggibile.
         /// </summary>
         /// <param name="index">L'indice dell'item</param>
         /// <returns>Lo stream di lettura</returns>
         public TextReader OpenTextReader(int index) => new StringReader(text ?? "");
         #endregion
      }
   }
}
