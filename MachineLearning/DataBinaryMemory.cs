﻿using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace MachineLearning
{
   /// <summary>
   /// Classe per lo storage di dati di tipo binario in memoria
   /// </summary>
   [Serializable]
   public sealed partial class DataBinaryMemory : IDataStorage, IMultiStreamSource, ITimestamp, ITrainingData
   {
      #region Fields
      /// <summary>
      /// Sorgente per il caricamento
      /// </summary>
      [NonSerialized]
      private Source _source;
      /// <summary>
      /// Dati testuali
      /// </summary>
      private byte[] _binaryData;
      #endregion
      #region Properties
      /// <summary>
      /// Il numero di items
      /// </summary>
      int IMultiStreamSource.Count => (_source ??= new Source(this)).Count;
      /// <summary>
      /// Dati testuali
      /// </summary>
      public byte[] BinaryData { get => _binaryData; set { _binaryData = value; Timestamp = DateTime.UtcNow; } }
      /// <summary>
      /// Data e ora dell'oggetto
      /// </summary>
      public DateTime Timestamp { get; private set; }
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      public DataBinaryMemory() { }
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      /// <param name="source">Storage di dati</param>
      public DataBinaryMemory(MachineLearningContext ml, IDataStorage source)
      {
         SaveData(ml, source.LoadData(ml, default), default);  
      }
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      /// <param name="data">Dati</param>
      public DataBinaryMemory(MachineLearningContext ml, IDataView data) => SaveData(ml, data, default);
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      /// <param name="data">Dati</param>
      public DataBinaryMemory(MachineLearningContext ml, byte[] data)
      {
         BinaryData = data;
         SaveData(ml, LoadData(ml, default), default);
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
      /// <param name="opt">Opzioni di testo (non usato)</param>
      /// <param name="extra">Sorgenti extra di dati</param>
      /// <returns>L'accesso ai dati</returns>
      public IDataView LoadData(MachineLearningContext ml, TextLoader.Options opt = default, params IMultiStreamSource[] extra)
      {
         return ml.NET.Data.LoadFromBinary(_source ??= new Source(this, extra));
      }
      /// <summary>
      /// Carica i dati di training
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      /// <param name="opt">Opzioni di testo (non usato)</param>
      /// <param name="extra">Sorgenti extra di dati</param>
      /// <returns>L'accesso ai dati</returns>
      public IDataView LoadTrainingData(MachineLearningContext ml, TextLoader.Options opt = default, params IMultiStreamSource[] extra) => LoadData(ml, default, extra);
      /// <summary>
      /// Salva i dati
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      /// <param name="data">L'accesso ai dati</param>
      /// <param name="opt">Opzioni di testo (non usato)</param>
      /// <param name="schema">Commento contenente lo schema nei dati di tipo file testuali (ignorato negli altri)</param>
      /// <param name="extra">Eventuali altri stream di dati</param>
      public void SaveData(MachineLearningContext ml, IDataView data, TextLoader.Options opt = default, bool schema = false, params IMultiStreamSource[] extra)
      {
         lock (this) {
            // Data e ora
            var timestamp = DateTime.UtcNow;
            // Oggetto per la scrittura dei dati in memoria
            using var writer = new MemoryStream();
            ml.NET.Data.SaveAsBinary(data, writer, true);
            // Salva gli stream extra
            foreach (var item in extra) {
               data = ml.NET.Data.LoadFromBinary(item);
               ml.NET.Data.SaveAsBinary(data, writer, true);
            }
            BinaryData = writer.ToArray();
            Timestamp = timestamp;
         }
      }
      /// <summary>
      /// Salva i dati di training
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      /// <param name="data">L'accesso ai dati</param>
      /// <param name="opt">Opzioni di testo (non usato)</param>
      /// <param name="schema">Commento contenente lo schema nei dati di tipo file testuali (ignorato negli altri)</param>
      /// <param name="extra">Eventuali altri stream di dati</param>
      public void SaveTrainingData(MachineLearningContext ml, IDataView data, TextLoader.Options opt = default, bool schema = false, params IMultiStreamSource[] extra) => SaveData(ml, data, default, schema, extra);
      #endregion
   }

   /// <summary>
   /// La sorgente dei dati
   /// </summary>
   public partial class DataBinaryMemory
   {
      private class Source : IMultiStreamSource
      {
         #region Fields
         private readonly DataBinaryMemory _owner;
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
         public Source(DataBinaryMemory owner, params IMultiStreamSource[] extra)
         {
            _owner = owner;
            var indices = new List<(IMultiStreamSource Source, int Index)>
            {
               (Source: this, Index: 0)
            };
            foreach (var e in extra) {
               foreach (var ix in Enumerable.Range(0, e?.Count ?? 0))
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
            return new MemoryStream(_owner.BinaryData);
         }
         /// <summary>
         /// Apre l'item indicato e ne restituisce uno stream di stringhe leggibile.
         /// </summary>
         /// <param name="index">L'indice dell'item</param>
         /// <returns>Lo stream di lettura</returns>
         public TextReader OpenTextReader(int index) => index == 0 ? new StreamReader(new MemoryStream(_owner.BinaryData ?? Array.Empty<byte>())) : _total[index].Source.OpenTextReader(_total[index].Index);
         #endregion
      }
   }
}
