﻿using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace MachineLearning
{
   /// <summary>
   /// Classe per lo storage di dati di tipo testo in memoria
   /// </summary>
   [Serializable]
   public sealed partial class DataStorageTextMemory : IDataStorage, IDataTextProvider, IMultiStreamSource, IDataTextOptionsProvider
   {
      #region Fields
      /// <summary>
      /// Sorgente per il caricamento
      /// </summary>
      [NonSerialized]
      private Source _source;
      #endregion
      #region Properties
      /// <summary>
      /// Il numero di items
      /// </summary>
      int IMultiStreamSource.Count => (_source ??= new Source(this)).Count;
      /// <summary>
      /// Dati testuali
      /// </summary>
      public string TextData { get; set; }
      /// <summary>
      /// Configurazione dei dati
      /// </summary>
      public TextLoaderOptions TextOptions { get; set; }
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      public DataStorageTextMemory() : this(null) { }
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="columns"></param>
      /// <param name="separator"></param>
      /// <param name="labelColumnName"></param>
      /// <param name="allowQuoting"></param>
      public DataStorageTextMemory(IEnumerable<string> columns = null, char separator = ',', string labelColumnName = "Label", bool allowQuoting = true)
      {
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

      string IMultiStreamSource.GetPathOrNull(int index) => (_source ??= new Source(this)).GetPathOrNull(index);
      /// <summary>
      /// Apre l'item indicato e ne restituisce uno stream leggibile.
      /// </summary>
      /// <param name="index">L'indice dell'item</param>
      /// <returns>Lo stream di lettura</returns>
      Stream IMultiStreamSource.Open(int index) => (_source ??= new Source(this)).Open(index);
      /// <summary>
      /// Apre l'item indicato e ne restituisce uno stream di stringhe leggibile.
      /// </summary>
      /// <param name="index">L'indice dell'item</param>
      /// <returns>Lo stream di lettura</returns>
      TextReader IMultiStreamSource.OpenTextReader(int index) => (_source ??= new Source(this)).OpenTextReader(index);
      /// <summary>
      /// Carica i dati
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      /// <param name="extra">Sorgenti extra di dati</param>
      /// <returns>L'accesso ai dati</returns>
      public IDataView LoadData(MachineLearningContext ml, params IMultiStreamSource[] extra)
      {
         return ml.NET.Data.CreateTextLoader(TextOptions ?? new TextLoader.Options()).Load(_source ??= new Source(this, extra));
      }
      /// <summary>
      /// Salva i dati
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      /// <param name="data">L'accesso ai dati</param>
      /// <param name="schema">Commento contenente lo schema nei dati di tipo file testuali (ignorato negli altri)</param>
      /// <param name="extra">Eventuali altri stream di dati</param>
      public void SaveData(MachineLearningContext ml, IDataView data, bool schema = false, params IMultiStreamSource[] extra)
      {
         // Oggetto per la scrittura dei dati in memoria
         using var writer = new MemoryStream();
         // Opzioni
         var opt = TextOptions ?? new TextLoader.Options();
         // Separatore di colonne
         var separator = opt.Separators?.FirstOrDefault() ?? '\t';
         separator = separator != default ? separator : '\t';
         // Salva come testo i dati
         ml.NET.Data.SaveAsText(
            data: data,
            stream: writer,
            separatorChar: separator,
            headerRow: opt.HasHeader,
            schema: false,
            keepHidden: false,
            forceDense: false);
         // Salva gli stream extra
         foreach (var item in extra) {
            var loader = ml.NET.Data.CreateTextLoader(opt);
            data = loader.Load(item);
            ml.NET.Data.SaveAsText(
               data: data,
               stream: writer,
               separatorChar: separator,
               headerRow: opt.HasHeader,
               schema: schema,
               keepHidden: false,
               forceDense: false);
         }
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
   public partial class DataStorageTextMemory
   {
      private class Source : IMultiStreamSource
      {
         #region Fields
         private readonly DataStorageTextMemory _owner;
         /// <summary>
         /// Sorgenti ed indici
         /// </summary>
         private readonly (IMultiStreamSource Source, int Index)[] _total;
         #endregion
         #region Properties
         /// <summary>
         /// Il numero di items
         /// </summary>
         public int Count => _total.Length;
         #endregion
         #region Methods
         /// <summary>
         /// Costruttore
         /// </summary>
         /// <param name="owner">Oggetto di appartenenza</param>
         /// <param name="extra">Sorgenti extra di dati</param>
         public Source(DataStorageTextMemory owner, params IMultiStreamSource[] extra)
         {
            _owner = owner;
            var indices = new List<(IMultiStreamSource Source, int Index)>
            {
               (Source: this, Index: 0)
            };
            foreach (var e in extra) {
               foreach (var ix in Enumerable.Range(0, e.Count))
                  indices.Add((Source: e, Index: ix));
            }
            _total = indices.ToArray();
         }
         /// <summary>
         /// Restituisce una stringa rappresentante il "path" dello stream indicato da index. Potrebbe essere null.
         /// </summary>
         /// <param name="index">L'indice dell'item</param>
         /// <returns>Sempre null</returns>
         public string GetPathOrNull(int index) => index == 0 ? default : _total[index].Source.GetPathOrNull(_total[index].Index);
         /// <summary>
         /// Apre l'item indicato e ne restituisce uno stream leggibile.
         /// </summary>
         /// <param name="index">L'indice dell'item</param>
         /// <returns>Lo stream di lettura</returns>
         public Stream Open(int index)
         {
            if (index > 0)
               return _total[index].Source.Open(_total[index].Index);
            var memoryStream = new MemoryStream();
            using var writer = new StreamWriter(memoryStream);
            writer.Write(_owner.TextData ?? "");
            memoryStream.Position = 0;
            return memoryStream;
         }
         /// <summary>
         /// Apre l'item indicato e ne restituisce uno stream di stringhe leggibile.
         /// </summary>
         /// <param name="index">L'indice dell'item</param>
         /// <returns>Lo stream di lettura</returns>
         public TextReader OpenTextReader(int index) => index == 0 ? new StringReader(_owner.TextData ?? "") : _total[index].Source.OpenTextReader(_total[index].Index);
         #endregion
      }
   }
}