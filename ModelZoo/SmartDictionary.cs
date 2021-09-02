using MachineLearning.Data;
using MachineLearning.Model;
using Microsoft.ML;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Threading;

namespace MachineLearning.ModelZoo
{
   /// <summary>
   /// Dizionario intelligente
   /// </summary>
   [Serializable]
   public partial class SmartDictionary<TValue> : ModelZooBase<SmartDictionary<TValue>.Mdl>, IDictionary<string, TValue>
   {
      #region Fields
      /// <summary>
      /// Dizionario interno
      /// </summary>
      private readonly Dictionary<string, TValue> internalDictionary = new();
      #endregion
      #region Properties
      /// <summary>
      /// Numero elementi
      /// </summary>
      public int Count => ((ICollection<KeyValuePair<string, TValue>>)internalDictionary).Count;
      /// <summary>
      /// Stato di sola lettura
      /// </summary>
      public bool IsReadOnly => ((ICollection<KeyValuePair<string, TValue>>)internalDictionary).IsReadOnly;
      /// <summary>
      /// Chiavi
      /// </summary>
      public ICollection<string> Keys => ((IDictionary<string, TValue>)internalDictionary).Keys;
      /// <summary>
      /// Inidicizzatore per similarita'
      /// </summary>
      public SimilarIndexer Similar => new(this);
      /// <summary>
      /// Indicizzatore per chiave
      /// </summary>
      /// <param name="key">Chiave</param>
      /// <returns>Il valore corrispondente</returns>
      public TValue this[string key]
      {
         get => ((IDictionary<string, TValue>)internalDictionary)[key];
         set
         {
            if (!internalDictionary.ContainsKey(key))
               Model.Add(key);
            ((IDictionary<string, TValue>)internalDictionary)[key] = value;
         }
      }
      /// <summary>
      /// Valori
      /// </summary>
      public ICollection<TValue> Values => ((IDictionary<string, TValue>)internalDictionary).Values;
      #endregion
      /// <summary>
      /// Costruttore
      /// </summary>
      public SmartDictionary() : this(Array.Empty<KeyValuePair<string, TValue>>()) { }
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="dictionary">Dizionario di inizializzazione</param>
      public SmartDictionary(IDictionary<string, TValue> dictionary) : this((IEnumerable<KeyValuePair<string, TValue>>)dictionary) { }
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="collection">Collezione di chiavi / valori</param>
      public SmartDictionary(IEnumerable<KeyValuePair<string, TValue>> collection)
      {
         internalDictionary = new(collection);
         Model = new();
         Model.AddRange(internalDictionary.Keys);
      }
      /// <summary>
      /// Aggiunge una coppia chiave valore
      /// </summary>
      /// <param name="key"></param>
      /// <param name="value"></param>
      public void Add(string key, TValue value)
      {
         ((IDictionary<string, TValue>)internalDictionary).Add(key, value);
         Model.Add(key);
      }
      /// <summary>
      /// Aggiunge una coppia chiave valore
      /// </summary>
      /// <param name="item"></param>
      public void Add(KeyValuePair<string, TValue> item)
      {
         ((ICollection<KeyValuePair<string, TValue>>)internalDictionary).Add(item);
         Model.Add(item.Key);
      }
      /// <summary>
      /// Cancella il dizionario
      /// </summary>
      public void Clear()
      {
         ((ICollection<KeyValuePair<string, TValue>>)internalDictionary).Clear();
         Model.ClearModel();
         ((IDataStorageProvider)Model).DataStorage.SaveData(Model.Context, DataViewGrid.Create(Model, ((IInputSchema)Model).InputSchema));
      }
      /// <summary>
      /// Verifica se il dizionario contiene una coppia chiave / valore
      /// </summary>
      /// <param name="item">Coppia da verificare</param>
      /// <returns></returns>
      public bool Contains(KeyValuePair<string, TValue> item) => ((ICollection<KeyValuePair<string, TValue>>)internalDictionary).Contains(item);
      /// <summary>
      /// Verifica se il dizionario contiene una chiave
      /// </summary>
      /// <param name="key">Chiave da verificare</param>
      /// <returns>true se la chiave e' presente</returns>
      public bool ContainsKey(string key) => ((IDictionary<string, TValue>)internalDictionary).ContainsKey(key);
      /// <summary>
      /// Copia il contenuto in un array di coppie chiave / valore
      /// </summary>
      /// <param name="array">Array di destinazione</param>
      /// <param name="arrayIndex">Indice di partenza nell'array</param>
      public void CopyTo(KeyValuePair<string, TValue>[] array, int arrayIndex) => ((ICollection<KeyValuePair<string, TValue>>)internalDictionary).CopyTo(array, arrayIndex);
      /// <summary>
      /// Restituisce l'enumeratore di coppie chiave / valore
      /// </summary>
      /// <returns>L'enumeratore</returns>
      public IEnumerator<KeyValuePair<string, TValue>> GetEnumerator() => ((IEnumerable<KeyValuePair<string, TValue>>)internalDictionary).GetEnumerator();
      /// <summary>
      /// Restituisce la chiave, contenuta nel dizionario, piu' simile a quella specificata
      /// </summary>
      /// <param name="key">La chiave</param>
      /// <param name="cancellation">Eventuale token di cancellazione</param>
      /// <returns>La chiave piu' simile</returns>
      public string GetSimilarKey(string key, CancellationToken cancellation = default) => Model.GetPredictionData(new[] { "", key }, cancellation).ToDataViewGrid()[0]["PredictedLabel"];
      /// <summary>
      /// Restituisce l'enumeratore
      /// </summary>
      /// <returns>L'enumeratore</returns>
      IEnumerator IEnumerable.GetEnumerator() => ((IEnumerable)internalDictionary).GetEnumerator();
      /// <summary>
      /// Rimuove un elemento per chiave
      /// </summary>
      /// <param name="key">Chiave di rimozione</param>
      /// <returns>true se l'elemento e' stato rimosso</returns>
      public bool Remove(string key)
      {
         if (((IDictionary<string, TValue>)internalDictionary).Remove(key)) {
            Model.ClearModel();
            ((IDataStorageProvider)Model).DataStorage.SaveData(Model.Context, DataViewGrid.Create(Model, ((IInputSchema)Model).InputSchema));
            foreach (var k in Keys)
               Model.Add(k);
         }
         return false;
      }
      /// <summary>
      /// Rimuove una coppia chiave / valore
      /// </summary>
      /// <param name="item">La coppia</param>
      /// <returns>true se la coppia e' stata eliminata</returns>
      public bool Remove(KeyValuePair<string, TValue> item)
      {
         if (((ICollection<KeyValuePair<string, TValue>>)internalDictionary).Remove(item)) {
            Model.ClearModel();
            ((IDataStorageProvider)Model).DataStorage.SaveData(Model.Context, DataViewGrid.Create(Model, ((IInputSchema)Model).InputSchema));
            foreach (var k in Keys)
               Model.Add(k);
            return true;
         }
         return false;
      }
      /// <summary>
      /// Prova ad estrarre un valore dal dizionario
      /// </summary>
      /// <param name="key">Chiave di estrazione</param>
      /// <param name="value">Valore estratto</param>
      /// <returns>true se la chiave era presente nel dizionario</returns>
      public bool TryGetValue(string key, [MaybeNullWhen(false)] out TValue value) => ((IDictionary<string, TValue>)internalDictionary).TryGetValue(key, out value);
   }

   public partial class SmartDictionary<TValue> // Mdl
   {
      /// <summary>
      /// Modello di interpretazione delle chiavi
      /// </summary>
      [Serializable]
      public class Mdl :
         MulticlassModelBase,
         IDataStorageProvider,
         IInputSchema,
         IModelAutoCommit,
         IModelAutoSave,
         IModelStorageProvider,
         IModelTrainerProvider,
         ITrainingStorageProvider
      {
         #region Fields
         /// <summary>
         /// Pipes
         /// </summary>
         [NonSerialized]
         private ModelPipes pipes;
         #endregion
         #region Properties
         /// <summary>
         /// Storage di dati
         /// </summary>
         IDataStorage IDataStorageProvider.DataStorage { get; } = new DataStorageBinaryMemory();
         /// <summary>
         /// Schema di input del modello
         /// </summary>
         DataSchema IInputSchema.InputSchema { get; } = DataViewSchemaBuilder.Build((Name: "Label", typeof(string)), (Name: "Text", typeof(string)));
         /// <summary>
         /// Abilitazione all'autocommit dei dati
         /// </summary>
         bool IModelAutoCommit.ModelAutoCommit => true;
         /// <summary>
         /// Abilitazione all'auto save del modello
         /// </summary>
         bool IModelAutoSave.ModelAutoSave => true;
         /// <summary>
         /// Storage del modello
         /// </summary>
         IModelStorage IModelStorageProvider.ModelStorage { get; } = new ModelStorageMemory();
         /// <summary>
         /// Trainer del modello
         /// </summary>
         IModelTrainer IModelTrainerProvider.ModelTrainer { get; } = new ModelTrainerStandard();
         /// <summary>
         /// Dati di training
         /// </summary>
         IDataStorage ITrainingStorageProvider.TrainingStorage { get; } = new DataStorageBinaryMemory();
         #endregion
         #region Methods
         /// <summary>
         /// Costruttore
         /// </summary>
         public Mdl() : base(new MachineLearningContext()) { }
         /// <summary>
         /// Aggiunge una chiave al modello
         /// </summary>
         /// <param name="key">Chiave</param>
         internal void Add(string key)
         {
            var data = DataViewGrid.Create(this, ((IInputSchema)this).InputSchema);
            data.Add(key, key);
            StopTraining();
            AddTrainingData(data, false);
         }
         /// <summary>
         /// Aggiunge una chiave al modello
         /// </summary>
         /// <param name="keys">Chiavi</param>
         internal void AddRange(IEnumerable<string> keys)
         {
            var data = DataViewGrid.Create(this, ((IInputSchema)this).InputSchema);
            foreach (var key in keys)
               data.Add(key, key);
            StopTraining();
            AddTrainingData(data, false);
         }
         /// <summary>
         /// Funzione di dispose
         /// </summary>
         /// <param name="disposing">Indicatore di dispose da codice</param>
         protected sealed override void Dispose(bool disposing)
         {
            base.Dispose(disposing);
            try {
               pipes?.Dispose();
            }
            catch (Exception exc) {
               Trace.WriteLine(exc);
            }
            pipes = null;
         }
         /// <summary>
         /// Restituisce le pipes
         /// </summary>
         /// <returns>Le pipe di apprendimento</returns>
         public sealed override ModelPipes GetPipes()
         {
            return pipes ??= new()
            {
               Trainer =
                  Context.Transforms.Text.NormalizeText("Text")
                  .Append(Context.Transforms.Text.TokenizeIntoWords("Text", null, new[] { '_', ',', ':', '[', ']', ' ' }))
                  .Append(Context.Transforms.Text.FeaturizeText("Text", "Text"))
                  .Append(Context.Transforms.Conversion.MapValueToKey("Label"))
                  .Append(Context.MulticlassClassification.Trainers.NaiveBayes("Label", "Text"))
                  .Append(Context.Transforms.Conversion.MapKeyToValue("PredictedLabel"))
            };
         }
         #endregion
      }
   }

   public partial class SmartDictionary<TValue> // Similarity
   {
      /// <summary>
      /// Indicizzatore per similarita' di chiave
      /// </summary>
      public class SimilarIndexer
      {
         #region Fields
         /// <summary>
         /// Oggetto di appartenenza
         /// </summary>
         private readonly SmartDictionary<TValue> owner;
         #endregion
         #region Properties
         /// <summary>
         /// Indicizzatore per chiave
         /// </summary>
         /// <param name="key">La chiave</param>
         /// <returns>Il valore corrispondente alla chiave piu' simile a quella richiesta</returns>
         public TValue this[string key]
         {
            get
            {
               if (!owner.TryGetValue(key, out var value))
                  value = owner[owner.GetSimilarKey(key)];
               return value;
            }
            set => owner[key] = value;
         }
         #endregion
         #region Methods
         /// <summary>
         /// Costruttore
         /// </summary>
         /// <param name="owner">Oggetto di appartenenza</param>
         public SimilarIndexer(SmartDictionary<TValue> owner) => this.owner = owner;
         #endregion
      }
   }
}
