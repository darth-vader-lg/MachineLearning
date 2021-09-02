using MachineLearning.Data;
using Microsoft.ML.Runtime;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Threading;

namespace MachineLearning.Model
{
   /// <summary>
   /// Catena di modelli
   /// </summary>
   public class ModelChain : IDataTransformer, IList<IDataTransformer>
   {
      #region Fields
      /// <summary>
      /// Lista di transformers
      /// </summary>
      private readonly List<IDataTransformer> models = new();
      #endregion
      #region Properties
      /// <summary>
      /// Indicizzatore
      /// </summary>
      /// <param name="index">Indice</param>
      /// <returns>Il trasformer all'indice specificato</returns>
      public IDataTransformer this[int index] { get => ((IList<IDataTransformer>)models)[index]; set => ((IList<IDataTransformer>)models)[index] = value; }
      /// <summary>
      /// Numero di transformers
      /// </summary>
      public int Count => ((ICollection<IDataTransformer>)models).Count;
      /// <summary>
      /// Indicatore di lista readonly
      /// </summary>
      public bool IsReadOnly => ((ICollection<IDataTransformer>)models).IsReadOnly;
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      public ModelChain() { }
      /// <summary>
      /// Aggiunge un transformer in coda alla lista
      /// </summary>
      /// <param name="item">Il transformer</param>
      public void Add(IDataTransformer item) => ((ICollection<IDataTransformer>)models).Add(item);
      /// <summary>
      /// Cancella la lista di transformers
      /// </summary>
      public void Clear() => ((ICollection<IDataTransformer>)models).Clear();
      /// <summary>
      /// Verifica se la lista contiene un transformer
      /// </summary>
      /// <param name="item">Il transformer</param>
      /// <returns>true se e' contenuto</returns>
      public bool Contains(IDataTransformer item) => ((ICollection<IDataTransformer>)models).Contains(item);
      /// <summary>
      /// Copia la lista di transofrmers in un array
      /// </summary>
      /// <param name="array">Array di destinazione</param>
      /// <param name="arrayIndex">Indice di partenza nell'array</param>
      public void CopyTo(IDataTransformer[] array, int arrayIndex) => ((ICollection<IDataTransformer>)models).CopyTo(array, arrayIndex);
      /// <summary>
      /// Restituisce l'enumeratore
      /// </summary>
      /// <returns>L'enumeratore</returns>
      public IEnumerator<IDataTransformer> GetEnumerator() => ((IEnumerable<IDataTransformer>)models).GetEnumerator();
      /// <summary>
      /// Restituisce il task di previsione
      /// </summary>
      /// <param name="data">Riga di dati da usare per la previsione</param>
      /// <param name="cancellation">Eventule token di cancellazione attesa</param>
      /// <returns>La previsione</returns>
      public IDataAccess GetPredictionData(IEnumerable<object> data, CancellationToken cancellation = default)
      {
         if (Count < 1)
            return null;
         if (this[0] is not IInputSchema inputSchema)
            throw new InvalidOperationException("Cannot infer the input schema of the model chain");
         var dataViewGrid = DataViewGrid.Create(this[0] as IChannelProvider ?? MachineLearningContext.Default, inputSchema.InputSchema);
         dataViewGrid.Add(data.ToArray());
         var result = (IDataAccess)dataViewGrid;
         cancellation.ThrowIfCancellationRequested();
         for (var i = 0; i < Count; i++) {
            result = this[i].Transform(result, cancellation);
            cancellation.ThrowIfCancellationRequested();
         }
         return result;
      }
      /// <summary>
      /// Restituisce l'enumeratore
      /// </summary>
      /// <returns>L'enumeratore</returns>
      IEnumerator IEnumerable.GetEnumerator() => ((IEnumerable)models).GetEnumerator();
      /// <summary>
      /// Restituisce l'indice di un transformer nella lista o -1 se non presente
      /// </summary>
      /// <param name="item">Il tranformer</param>
      /// <returns>L'indice</returns>
      public int IndexOf(IDataTransformer item) => ((IList<IDataTransformer>)models).IndexOf(item);
      /// <summary>
      /// Inserisce un transformer in una posizione della lista
      /// </summary>
      /// <param name="index">Indice di inserimento</param>
      /// <param name="item">Il transformer</param>
      public void Insert(int index, IDataTransformer item) => ((IList<IDataTransformer>)models).Insert(index, item);
      /// <summary>
      /// Rimuove un transformer dalla lista
      /// </summary>
      /// <param name="item">Il transformer</param>
      /// <returns>true se rimosso</returns>
      public bool Remove(IDataTransformer item) => ((ICollection<IDataTransformer>)models).Remove(item);
      /// <summary>
      /// Rimuove un transformer ad una determinata posizione nella lista
      /// </summary>
      /// <param name="index">Indice nella lista</param>
      public void RemoveAt(int index) => ((IList<IDataTransformer>)models).RemoveAt(index);
      /// <summary>
      /// Trasforma i dati
      /// </summary>
      /// <param name="data">Dati in ingresso</param>
      /// <param name="cancellation">Eventuale token di cancellazione</param>
      /// <returns>I dati trasformati</returns>
      public IDataAccess Transform(IDataAccess data, CancellationToken cancellation = default)
      {
         foreach (var transformer in this) {
            data = transformer.Transform(data, cancellation);
            cancellation.ThrowIfCancellationRequested();
         }
         return data;
      }
      #endregion
   }
}
