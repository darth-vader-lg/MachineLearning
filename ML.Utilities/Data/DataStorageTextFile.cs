﻿using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace ML.Utilities.Data
{
   /// <summary>
   /// Classe per lo storage di dati di tipo file di testo
   /// </summary>
   [Serializable]
   public sealed partial class DataStorageTextFile : IDataStorage, IMultiStreamSource, ITextOptionsProvider
   {
      #region Fields
      /// <summary>
      /// Sorgente per il caricamento
      /// </summary>
      private Source source;
      #endregion
      #region Properties
      /// <summary>
      /// Il numero di items
      /// </summary>
      int IMultiStreamSource.Count => (source ??= new Source(null)).Count;
      /// <summary>
      /// Path del file
      /// </summary>
      public string FilePath { get; private set; }
      /// <summary>
      /// Configurazione dei dati
      /// </summary>
      public TextLoader.Options TextOptions { get; set; }
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="filePath">Path del file</param>
      public DataStorageTextFile(string filePath) : this(filePath, null) { }
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="filePath">Path del file</param>
      /// <param name="columns"></param>
      /// <param name="separator"></param>
      /// <param name="labelColumnName"></param>
      /// <param name="allowQuoting"></param>
      public DataStorageTextFile(string filePath,  IEnumerable<string> columns = null, char separator = ',', string labelColumnName = "Label", bool allowQuoting = true)
      {
         this.FilePath = filePath;
         TextOptions = new TextLoader.Options
         {
            AllowQuoting = allowQuoting,
            AllowSparse = false,
            Separators = new[] { separator },
            Columns = columns != default ?
            columns.Select((c, i) => new TextLoader.Column(c == labelColumnName ? "Label" : c, DataKind.String, i)).ToArray() :
            new[]
            {
               new TextLoader.Column("Label", DataKind.String, 0),
               new TextLoader.Column("Text", DataKind.String, 1),
            }
         };
      }
      /// <summary>
      /// Restituisce una stringa rappresentante il "path" dello stream indicato da index. Potrebbe essere null.
      /// </summary>
      /// <param name="index">L'indice dell'item</param>
      /// <returns>Il path</returns>

      string IMultiStreamSource.GetPathOrNull(int index) => (source ??= new Source(FilePath)).GetPathOrNull(index);
      /// <summary>
      /// Apre l'item indicato e ne restituisce uno stream leggibile.
      /// </summary>
      /// <param name="index">L'indice dell'item</param>
      /// <returns>Lo stream di lettura</returns>
      Stream IMultiStreamSource.Open(int index) => (source ??= new Source(FilePath)).Open(index);
      /// <summary>
      /// Apre l'item indicato e ne restituisce uno stream di stringhe leggibile.
      /// </summary>
      /// <param name="index">L'indice dell'item</param>
      /// <returns>Lo stream di lettura</returns>
      TextReader IMultiStreamSource.OpenTextReader(int index) => (source ??= new Source(FilePath)).OpenTextReader(index);
      /// <summary>
      /// Carica i dati
      /// </summary>
      /// <param name="mlContext">Contesto di machine learning</param>
      /// <param name="extra">Sorgenti extra di dati</param>
      /// <returns>L'accesso ai dati</returns>
      public IDataView LoadData(MLContext mlContext, params IMultiStreamSource[] extra)
      {
         return mlContext.Data.CreateTextLoader(TextOptions ?? new TextLoader.Options()).Load(source = new Source(FilePath, extra));
      }
      /// <summary>
      /// Salva i dati
      /// </summary>
      /// <param name="mlContext">Contesto di machine learning</param>
      /// <param name="data">L'accesso ai dati</param>
      public void SaveData(MLContext mlContext, IDataView data)
      {
         // Oggetto per la scrittura dei dati in memoria
         using var writer = File.OpenWrite(FilePath);
         // Opzioni
         var opt = TextOptions ?? new TextLoader.Options();
         // Separatore di colonne
         var separator = opt.Separators?.FirstOrDefault() ?? '\t';
         separator = separator != default ? separator : '\t';
         // Salva come testo i dati
         mlContext.Data.SaveAsText(
            data: data,
            stream: writer,
            separatorChar: separator,
            headerRow: opt.HasHeader,
            schema: true/*@@@**/,
            keepHidden: false,
            forceDense: false);
      }
      #endregion
   }

   /// <summary>
   /// La sorgente dei dati
   /// </summary>
   public partial class DataStorageTextFile
   {
      private class Source : IMultiStreamSource
      {
         #region Fields
         /// <summary>
         /// Sorgenti ed indici
         /// </summary>
         private readonly (IMultiStreamSource Source, int Index)[] total;
         /// <summary>
         /// Testo
         /// </summary>
         private readonly string filePath;
         #endregion
         #region Properties
         /// <summary>
         /// Il numero di items
         /// </summary>
         public int Count => total.Length;
         #endregion
         #region Methods
         /// <summary>
         /// Costruttore
         /// </summary>
         /// <param name="filePath">Path del file</param>
         /// <param name="extra">Sorgenti extra di dati</param>
         public Source(string filePath, params IMultiStreamSource[] extra)
         {
            this.filePath = filePath;
            var indices = new List<(IMultiStreamSource Source, int Index)>();
            indices.Add((Source: this, Index: 0));
            foreach (var e in extra) {
               foreach (var ix in Enumerable.Range(0, e.Count))
                  indices.Add((Source: e, Index: ix));
            }
            total = indices.ToArray();
         }
         /// <summary>
         /// Restituisce una stringa rappresentante il "path" dello stream indicato da index. Potrebbe essere null.
         /// </summary>
         /// <param name="index">L'indice dell'item</param>
         /// <returns>Sempre null</returns>
         public string GetPathOrNull(int index) => index == 0 ? filePath : total[index].Source.GetPathOrNull(total[index].Index);
         /// <summary>
         /// Apre l'item indicato e ne restituisce uno stream leggibile.
         /// </summary>
         /// <param name="index">L'indice dell'item</param>
         /// <returns>Lo stream di lettura</returns>
         public Stream Open(int index) => index == 0 ? File.OpenRead(filePath) : total[index].Source.Open(total[index].Index);
         /// <summary>
         /// Apre l'item indicato e ne restituisce uno stream di stringhe leggibile.
         /// </summary>
         /// <param name="index">L'indice dell'item</param>
         /// <returns>Lo stream di lettura</returns>
         public TextReader OpenTextReader(int index) => index == 0 ? new StreamReader(filePath) : total[index].Source.OpenTextReader(total[index].Index);
         #endregion
      }
   }
}
