using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace MachineLearning.Model
{
   /// <summary>
   /// Modello composito
   /// </summary>
   public class CompositeModel : List<ITransformer>, ITransformer
   {
      #region Delegates
      /// <summary>
      /// Delegato alla restituzione di uno stream
      /// </summary>
      /// <param name="index">Indice del transformer</param>
      /// <param name="write">Modelaita' di apertura (lettura/scrittura)</param>
      /// <returns></returns>
      public delegate Stream StreamGetter(int index, bool write);
      #endregion
      #region Fields
      /// <summary>
      /// Contesto di machine learning
      /// </summary>
      private readonly MachineLearningContext _ml;
      /// <summary>
      /// Delegato alla restituzione degli stream per la persistenza
      /// </summary>
      private readonly StreamGetter _streamGetter;
      #endregion
      #region Properties
      /// <summary>
      /// Schema di input
      /// </summary>
      public DataViewSchema Schema { get; set; }
      /// <summary>
      /// Stato di mapper Row to Row
      /// </summary>
      public bool IsRowToRowMapper => this.Any(t => t?.IsRowToRowMapper ?? false);
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <param name="streamGetter">Eventuale delegato alla restituzione degli stream per la persistenza</param>
      public CompositeModel(IMachineLearningContextProvider context, StreamGetter streamGetter = null)
      {
         MachineLearningContext.CheckMLNET(context, nameof(context));
         _ml = context.ML;
         _streamGetter = streamGetter;
      }
      /// <summary>
      /// Restituisce lo schema di output
      /// </summary>
      /// <param name="inputSchema">Schema di input</param>
      /// <returns>Lo schema di output</returns>
      public DataViewSchema GetOutputSchema(DataViewSchema inputSchema)
      {
         var result = inputSchema;
         ForEach(t => result = t == null ? result : t.GetOutputSchema(result));
         return result;
      }
      /// <summary>
      /// restituisce il mapper riga a riga
      /// </summary>
      /// <param name="inputSchema"></param>
      /// <returns></returns>
      public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
      {
         IRowToRowMapper rtrm = null;
         ForEach(t =>
         {
            if (t != null) {
               rtrm = t.GetRowToRowMapper(inputSchema);
               inputSchema = rtrm.OutputSchema;
            }
         });
         return rtrm;
      }
      /// <summary>
      /// Carica i modelli
      /// </summary>
      /// <param name="streams">Streams dei modelli</param>
      /// <returns>I modelli</returns>
      public CompositeModel Load()
      {
         _ml.NET.Check(_streamGetter != null, "No streams specified");
         Clear();
         Schema = null;
         for (var stream = _streamGetter(Count, false); stream != null; stream = _streamGetter(Count, false)) {
            Add(_ml.NET.Model.Load(stream, out var schema));
            Schema ??= schema;
         }
         return this;
      }
      /// <summary>
      /// Effettua la trasformazione dei dati
      /// </summary>
      /// <param name="input">Dati di input</param>
      /// <returns>I dati trasformati</returns>
      public IDataView Transform(IDataView input)
      {
         ForEach(t => input = t == null ? input : t.Transform(input));
         return input;
      }
      /// <summary>
      /// Salva i modelli
      /// </summary>
      /// <param name="ctx">Contesto di salvataggio</param>
      public void Save(ModelSaveContext ctx)
      {
         _ml.NET.Check(_streamGetter != null, "No streams specified");
         Save(_streamGetter);
      }
      /// <summary>
      /// Salva i modelli
      /// </summary>
      /// <param name="streamGetter">Delegato alla restituzione degli stream per la persistenza</param>
      public void Save(StreamGetter streamGetter)
      {
         _ml.NET.CheckParam(streamGetter != null, nameof(streamGetter));
         for (var i = 0; i < Count; i++) {
            var stream = streamGetter(i, true);
            _ml.NET.CheckIO(stream != null, "Null stream writing the model");
            _ml.NET.Model.Save(this[i], i == 0 ? Schema : null, stream);
         }
      }
      #endregion
   }
}
